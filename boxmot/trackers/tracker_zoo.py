# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import importlib
import yaml

from boxmot.utils import CONFIGS, TRACKER_CONFIGS

REID_TRACKERS = ["deepsort"]

TRACKER_MAPPING = {
    "deepsort": "boxmot.trackers.deepsort.deepsort.DeepSort",
}


def get_tracker_config(tracker_type):
    """Resolve the configuration file path for a supported tracker."""
    filename = f"{tracker_type}.yaml"
    candidates = [
        TRACKER_CONFIGS / filename,
        CONFIGS / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def create_tracker(
    tracker_type,
    tracker_config=None,
    reid_weights=None,
    device=None,
    half=None,
    per_class=None,
    evolve_param_dict=None,
):
    """
    Build and initialize a tracker instance from YAML defaults or an override dict.

    Parameters
    ----------
    tracker_type:
        Canonical tracker name. Only ``deepsort`` is supported in this project.
    tracker_config:
        Optional path to a YAML configuration file.
    reid_weights, device, half:
        Runtime parameters forwarded to trackers that require appearance features.
    per_class:
        Whether to maintain class-specific tracker state.
    evolve_param_dict:
        Optional dictionary that overrides YAML defaults.
    """

    if tracker_type not in TRACKER_MAPPING:
        raise ValueError(
            f"For this graduation project, only 'deepsort' is supported. Got: {tracker_type}"
        )

    if evolve_param_dict is None:
        if tracker_config is None:
            tracker_config = get_tracker_config(tracker_type)
        with open(tracker_config, "r") as f:
            yaml_config = yaml.safe_load(f)
            tracker_args = {
                param: details["default"] for param, details in yaml_config.items()
            }
    else:
        tracker_args = evolve_param_dict.copy()

    tracker_args["per_class"] = per_class

    if tracker_type in REID_TRACKERS:
        tracker_args.update(
            {
                "reid_weights": reid_weights,
                "device": device,
                "half": half,
            }
        )

    module_path, class_name = TRACKER_MAPPING[tracker_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    tracker_class = getattr(module, class_name)

    tracker = tracker_class(**tracker_args)
    if hasattr(tracker, "model"):
        tracker.model.warmup()
    return tracker
