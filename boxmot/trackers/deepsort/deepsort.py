# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

from boxmot.motion.kalman_filters.aabb.xyah_kf import KalmanFilterXYAH
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.matching import fuse_motion, iou_distance, linear_assignment
from boxmot.utils.ops import tlwh2xyah, tlwh2xyxy, xyxy2tlwh


class TrackState:
    """Lifecycle states used by the DeepSORT track manager."""

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Detection:
    """Container for a detection box and its appearance embedding."""

    def __init__(self, det: np.ndarray, feature: np.ndarray):
        self.tlwh = xyxy2tlwh(det[:4]).astype(np.float32)
        self.conf = float(det[4])
        self.cls = float(det[5])
        self.det_ind = int(det[6])
        self.curr_feat = self._normalize(feature)

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        feat = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    @property
    def xyxy(self) -> np.ndarray:
        return tlwh2xyxy(self.tlwh).astype(np.float32)

    def to_xyah(self) -> np.ndarray:
        return tlwh2xyah(self.tlwh).astype(np.float32)


class Track:
    """Single target state tracked with a Kalman filter and feature history."""

    def __init__(
        self,
        detection: Detection,
        track_id: int,
        n_init: int,
        max_age: int,
        max_obs: int,
        nn_budget: int,
        ema_alpha: float,
        frame_id: int,
        kf: KalmanFilterXYAH,
    ):
        self.id = track_id
        self.track_id = track_id
        self.n_init = n_init
        self.max_age = max_age
        self.ema_alpha = ema_alpha

        self.mean, self.covariance = kf.initiate(detection.to_xyah())
        self.state = TrackState.Tentative

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.start_frame = frame_id
        self.birth_frame = frame_id

        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind

        self.curr_feat = None
        self.smooth_feat = None
        self.features = deque([], maxlen=nn_budget)
        self.history_observations = deque([], maxlen=max_obs)
        self._update_feature(detection.curr_feat)
        self.history_observations.append(self.to_xyxy())
        if self.n_init <= 1:
            self.state = TrackState.Confirmed

    def _update_feature(self, feature: np.ndarray) -> None:
        if feature is None or feature.size == 0:
            return
        feat = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.ema_alpha * self.smooth_feat + (1.0 - self.ema_alpha) * feat
            smooth_norm = np.linalg.norm(self.smooth_feat)
            if smooth_norm > 0:
                self.smooth_feat /= smooth_norm
        self.features.append(feat.copy())

    def predict(self, kf: KalmanFilterXYAH) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilterXYAH, detection: Detection) -> None:
        self.mean, self.covariance = kf.update(
            self.mean,
            self.covariance,
            detection.to_xyah(),
            confidence=detection.conf,
        )
        self._update_feature(detection.curr_feat)
        self.history_observations.append(self.to_xyxy())
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted

    def to_tlwh(self) -> np.ndarray:
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret.astype(np.float32)

    def to_xyxy(self) -> np.ndarray:
        return tlwh2xyxy(self.to_tlwh()).astype(np.float32)

    @property
    def xyxy(self) -> np.ndarray:
        return self.to_xyxy()

    def get_state(self) -> np.ndarray:
        return self.to_xyxy()


class DeepSort(BaseTracker):
    """DeepSORT tracker with appearance matching and IoU fallback association."""

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        max_dist: float = 0.2,
        max_iou_dist: float = 0.7,
        nn_budget: int = 100,
        mc_lambda: float = 0.98,
        ema_alpha: float = 0.9,
        **kwargs,
    ):
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="DeepSort", **kwargs)

        self.max_dist = max_dist
        self.max_iou_dist = max_iou_dist
        self.nn_budget = nn_budget
        self.mc_lambda = mc_lambda
        self.ema_alpha = ema_alpha
        self.n_init = self.min_hits

        self.kf = KalmanFilterXYAH()
        self.model = ReidAutoBackend(weights=reid_weights, device=device, half=half).model
        self._next_id = 1

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        """Update tracker state for one frame and return confirmed tracks."""
        self.check_inputs(dets, img, embs)
        self.frame_count += 1

        dets = np.asarray(dets, dtype=np.float32)
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)

        dets_with_index = np.hstack(
            [dets, np.arange(len(dets), dtype=np.float32).reshape(-1, 1)]
        )
        remain_inds = dets[:, 4] > self.det_thresh if dets.shape[0] > 0 else np.array([], dtype=bool)
        dets_with_index = dets_with_index[remain_inds]

        if dets_with_index.shape[0] == 0:
            det_features = np.empty((0, 0), dtype=np.float32)
        elif embs is not None:
            det_features = np.asarray(embs[remain_inds], dtype=np.float32)
        else:
            det_features = np.asarray(
                self.model.get_features(dets_with_index[:, :4], img),
                dtype=np.float32,
            )

        detections = [
            Detection(det, det_features[i])
            for i, det in enumerate(dets_with_index)
        ]

        for track in self.active_tracks:
            track.predict(self.kf)

        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.active_tracks[track_idx].update(self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            self.active_tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self.active_tracks = [track for track in self.active_tracks if not track.is_deleted()]

        outputs = []
        for track in self.active_tracks:
            if track.time_since_update > 0:
                continue
            if not track.is_confirmed() and self.frame_count > self.n_init:
                continue
            bbox = track.to_xyxy()
            outputs.append(
                np.array(
                    [
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                        track.id,
                        track.conf,
                        track.cls,
                        track.det_ind,
                    ],
                    dtype=np.float32,
                )
            )

        return np.vstack(outputs) if outputs else np.empty((0, 8), dtype=np.float32)

    def _match(self, detections):
        """Run appearance-first matching followed by IoU-based recovery."""
        confirmed_tracks = [
            i for i, track in enumerate(self.active_tracks) if track.is_confirmed()
        ]
        unconfirmed_tracks = [
            i for i, track in enumerate(self.active_tracks) if not track.is_confirmed()
        ]

        matches_a, unmatched_confirmed, unmatched_detections = self._matching_cascade(
            detections,
            confirmed_tracks,
        )

        iou_track_candidates = unconfirmed_tracks + [
            track_idx
            for track_idx in unmatched_confirmed
            if self.active_tracks[track_idx].time_since_update == 1
        ]
        unmatched_confirmed = [
            track_idx
            for track_idx in unmatched_confirmed
            if self.active_tracks[track_idx].time_since_update != 1
        ]

        matches_b, unmatched_iou_tracks, unmatched_detections = self._min_cost_matching(
            detections,
            iou_track_candidates,
            unmatched_detections,
            self.max_iou_dist,
            metric="iou",
        )

        matches = matches_a + matches_b
        unmatched_tracks = unmatched_confirmed + unmatched_iou_tracks
        return matches, unmatched_tracks, unmatched_detections

    def _matching_cascade(self, detections, track_indices):
        """Prioritize recently updated confirmed tracks during association."""
        unmatched_detections = list(range(len(detections)))
        matches = []

        for level in range(self.max_age):
            if not unmatched_detections:
                break
            level_tracks = [
                track_idx
                for track_idx in track_indices
                if self.active_tracks[track_idx].time_since_update == level + 1
            ]
            if not level_tracks:
                continue
            matches_l, _, unmatched_detections = self._min_cost_matching(
                detections,
                level_tracks,
                unmatched_detections,
                self.max_dist,
                metric="appearance",
            )
            matches.extend(matches_l)

        matched_track_ids = {track_idx for track_idx, _ in matches}
        unmatched_tracks = [
            track_idx for track_idx in track_indices if track_idx not in matched_track_ids
        ]
        return matches, unmatched_tracks, unmatched_detections

    def _min_cost_matching(
        self,
        detections,
        track_indices,
        detection_indices,
        threshold,
        metric,
    ):
        """Solve an assignment problem for the selected tracks and detections."""
        if not track_indices or not detection_indices:
            return [], list(track_indices), list(detection_indices)

        selected_tracks = [self.active_tracks[i] for i in track_indices]
        selected_detections = [detections[i] for i in detection_indices]

        if metric == "appearance":
            cost_matrix = self._gated_appearance_distance(selected_tracks, selected_detections)
        elif metric == "iou":
            cost_matrix = iou_distance(selected_tracks, selected_detections)
        else:
            raise ValueError(f"Unsupported matching metric: {metric}")

        cost_matrix = np.where(np.isfinite(cost_matrix), cost_matrix, threshold + 1e5)
        matches, unmatched_tracks, unmatched_detections = linear_assignment(
            cost_matrix,
            threshold,
        )

        match_pairs = [
            (track_indices[row], detection_indices[col]) for row, col in matches
        ]
        unmatched_track_ids = [track_indices[row] for row in unmatched_tracks]
        unmatched_detection_ids = [detection_indices[col] for col in unmatched_detections]
        return match_pairs, unmatched_track_ids, unmatched_detection_ids

    def _gated_appearance_distance(self, tracks, detections):
        """Fuse cosine appearance distance with motion-based gating."""
        cost_matrix = self._nn_cosine_distance(tracks, detections)
        return fuse_motion(
            self.kf,
            cost_matrix,
            tracks,
            detections,
            lambda_=self.mc_lambda,
        )

    @staticmethod
    def _nn_cosine_distance(tracks, detections):
        """Compute nearest-neighbor cosine distance against each track gallery."""
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        det_features = np.asarray([det.curr_feat for det in detections], dtype=np.float32)
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for row, track in enumerate(tracks):
            if track.features:
                gallery = np.asarray(track.features, dtype=np.float32)
            elif track.smooth_feat is not None:
                gallery = np.asarray([track.smooth_feat], dtype=np.float32)
            else:
                cost_matrix[row, :] = 1.0
                continue
            distances = cdist(gallery, det_features, metric="cosine")
            cost_matrix[row, :] = np.min(distances, axis=0)

        return cost_matrix

    def _initiate_track(self, detection: Detection) -> None:
        """Create a new tentative track from an unmatched detection."""
        track = Track(
            detection=detection,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            max_obs=self.max_obs,
            nn_budget=self.nn_budget,
            ema_alpha=self.ema_alpha,
            frame_id=self.frame_count,
            kf=self.kf,
        )
        self._next_id += 1
        self.active_tracks.append(track)
