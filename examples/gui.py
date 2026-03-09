import os
import queue
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

class TrackingGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("MOT Control Deck")
        self.master.geometry("1080x740")
        self.master.minsize(980, 700)

        self.process = None
        self.current_task = ""
        self.log_queue = queue.Queue()
        self.python_exec = self._resolve_python_exec()

        self.theme = {
            "bg": "#F2ECE3",
            "panel": "#FFF8EF",
            "panel_alt": "#F6EFE5",
            "border": "#D7C6B5",
            "text": "#132238",
            "muted": "#667085",
            "hero_bg": "#0F172A",
            "hero_panel": "#16253D",
            "hero_border": "#24405F",
            "hero_text": "#F8FAFC",
            "hero_muted": "#B7C4D6",
            "accent": "#0F766E",
            "accent_hover": "#115E59",
            "secondary": "#D97706",
            "secondary_hover": "#B45309",
            "danger": "#B91C1C",
            "danger_hover": "#991B1B",
            "ghost": "#334155",
            "ghost_hover": "#1F2937",
            "log_bg": "#0B1220",
            "log_fg": "#DCE7F3",
            "highlight": "#F59E0B",
        }

        self.status_var = tk.StringVar(value="Idle")
        self.hero_detector_var = tk.StringVar(value="YOLOV5 / DEEPSORT")
        self.hero_results_var = tk.StringVar(value="0 result folders")
        self.hero_workspace_var = tk.StringVar(value=str(ROOT / "runs"))
        self.tracking_runs_var = tk.StringVar(value="0")
        self.pipeline_runs_var = tk.StringVar(value="0")
        self.result_files_var = tk.StringVar(value="0")
        self.result_size_var = tk.StringVar(value="0 B")
        self.result_root_var = tk.StringVar(value=str(ROOT / "runs"))

        self._setup_style()
        self._build_ui()
        self._fill_defaults()
        self._bind_dynamic_state()
        self._refresh_result_summary()
        self._center_window()
        self._set_status("Idle")
        self._append_log(f"Python environment: {self.python_exec}\n")
        self._bring_to_front()

    def _resolve_python_exec(self) -> str:
        tracker_python = Path("/opt/homebrew/anaconda3/envs/tracker/bin/python")
        if tracker_python.exists():
            return str(tracker_python)
        return sys.executable

    def _center_window(self):
        self.master.update_idletasks()
        w = self.master.winfo_width()
        h = self.master.winfo_height()
        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.master.geometry(f"{w}x{h}+{x}+{y}")

    def _bring_to_front(self):
        self.master.deiconify()
        self.master.lift()
        self.master.focus_force()
        self.master.attributes("-topmost", True)
        self.master.after(250, lambda: self.master.attributes("-topmost", False))

    def _setup_style(self):
        self.master.configure(bg=self.theme["bg"])
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Root.TFrame", background=self.theme["bg"])
        style.configure("Card.TLabelframe", background=self.theme["panel"], bordercolor=self.theme["border"], borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background=self.theme["panel"], foreground=self.theme["text"], font=("Avenir Next", 11, "bold"))
        style.configure("Card.TFrame", background=self.theme["panel"])
        style.configure("TLabel", background=self.theme["bg"], foreground=self.theme["text"], font=("Avenir Next", 10))
        style.configure("Panel.TLabel", background=self.theme["panel"], foreground=self.theme["text"], font=("Avenir Next", 10))
        style.configure("Hint.TLabel", background=self.theme["panel"], foreground=self.theme["muted"], font=("Avenir Next", 9))
        style.configure("Field.TLabel", background=self.theme["panel"], foreground=self.theme["text"], font=("Avenir Next", 10))
        style.configure("Header.TLabel", background=self.theme["bg"], foreground=self.theme["text"], font=("Avenir Next", 16, "bold"))
        style.configure("SubHeader.TLabel", background=self.theme["bg"], foreground=self.theme["muted"], font=("Avenir Next", 9))
        style.configure("TEntry", padding=7, fieldbackground=self.theme["panel_alt"], foreground=self.theme["text"])
        style.configure("TCombobox", padding=6, fieldbackground=self.theme["panel_alt"], foreground=self.theme["text"])
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.theme["panel_alt"])],
            selectbackground=[("readonly", self.theme["panel_alt"])],
            selectforeground=[("readonly", self.theme["text"])],
        )
        style.configure("TCheckbutton", background=self.theme["panel"], foreground=self.theme["text"], font=("Avenir Next", 10))
        style.map("TCheckbutton", background=[("active", self.theme["panel"])])

        self._configure_button_style(style, "Primary.Action.TButton", self.theme["accent"], self.theme["accent_hover"])
        self._configure_button_style(style, "Secondary.Action.TButton", self.theme["secondary"], self.theme["secondary_hover"])
        self._configure_button_style(style, "Danger.Action.TButton", self.theme["danger"], self.theme["danger_hover"])
        self._configure_button_style(style, "Ghost.Action.TButton", self.theme["ghost"], self.theme["hero_bg"])

    def _configure_button_style(self, style: ttk.Style, name: str, color: str, hover: str):
        style.configure(name, font=("Avenir Next", 10, "bold"), padding=(12, 8))
        style.map(
            name,
            background=[("active", hover), ("!disabled", color)],
            foreground=[("!disabled", "#FFFFFF"), ("disabled", "#D1D5DB")],
        )

    def _build_ui(self):
        root = ttk.Frame(self.master, style="Root.TFrame")
        root.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        self._build_hero(root)
        self._build_toolbar(root)

        content = ttk.Frame(root, style="Root.TFrame")
        content.pack(fill=tk.X, expand=False)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)

        form = ttk.LabelFrame(content, text="Tracking Console", style="Card.TLabelframe")
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        form.columnconfigure(1, weight=1)
        pad = {"padx": 10, "pady": 6}
        row = 0

        ttk.Label(form, text="Source preset", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.source_preset_var = tk.StringVar()
        self.source_preset = ttk.Combobox(
            form,
            textvariable=self.source_preset_var,
            state="readonly",
            values=["Webcam (0)", "Demo video 1 (test_video.mp4)", "Demo video 2 (test_video_2.mp4)", "Custom"],
            width=28,
        )
        self.source_preset.grid(row=row, column=1, sticky="ew", **pad)
        self.source_preset.bind("<<ComboboxSelected>>", self._on_source_preset_change)

        row += 1
        ttk.Label(form, text="Source", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.source_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.source_var, width=62).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(form, text="Browse Video", style="Ghost.Action.TButton", command=self._pick_source).grid(row=row, column=2, sticky="e", padx=(0, 10), pady=6)

        row += 1
        ttk.Label(form, text="YOLO version", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.yolo_version_var = tk.StringVar()
        self.yolo_version_combo = ttk.Combobox(form, textvariable=self.yolo_version_var, state="readonly", values=["yolov5", "yolov8"], width=18)
        self.yolo_version_combo.grid(row=row, column=1, sticky="w", **pad)
        self.yolo_version_combo.bind("<<ComboboxSelected>>", self._on_yolo_version_change)

        row += 1
        ttk.Label(form, text="YOLO weights", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.yolo_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.yolo_var, width=68).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(form, text="Browse Weights", style="Ghost.Action.TButton", command=self._pick_yolo_file).grid(row=row, column=2, sticky="e", padx=(0, 10), pady=6)

        row += 1
        ttk.Label(form, text="ReID weights", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.reid_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.reid_var, width=68).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(form, text="Browse Weights", style="Ghost.Action.TButton", command=lambda: self._pick_file(self.reid_var)).grid(row=row, column=2, sticky="e", padx=(0, 10), pady=6)

        row += 1
        ttk.Label(form, text="Confidence threshold", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.conf_var = tk.StringVar(value="0.5")
        ttk.Entry(form, textvariable=self.conf_var, width=16).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(form, text="IoU threshold", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.iou_var = tk.StringVar(value="0.5")
        ttk.Entry(form, textvariable=self.iou_var, width=16).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(form, text="YOLOv8 input size (imgsz)", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.imgsz_var = tk.StringVar(value="640")
        ttk.Entry(form, textvariable=self.imgsz_var, width=16).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(form, text="YOLOv8 max detections (max_det)", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.max_det_var = tk.StringVar(value="150")
        ttk.Entry(form, textvariable=self.max_det_var, width=16).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        self.person_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="YOLOv8 person-only mode (class=0)", variable=self.person_only_var).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        self.show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Show OpenCV preview window", variable=self.show_var).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(form, text="Note", style="Field.TLabel").grid(row=row, column=0, sticky="nw")
        ttk.Label(
            form,
            text="The detector version is aligned with the selected weight file automatically. Choosing a yolov8 weight switches the GUI to YOLOv8.",
            style="Hint.TLabel",
            wraplength=480,
            justify=tk.LEFT,
        ).grid(row=row, column=1, sticky="w", **pad)

        right_col = ttk.Frame(content, style="Root.TFrame")
        right_col.grid(row=0, column=1, sticky="nsew")

        self._build_results_workspace(right_col)
        self._build_eval_form(right_col)

        log_wrap = ttk.LabelFrame(root, text="Mission Log", style="Card.TLabelframe")
        log_wrap.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_box = tk.Text(
            log_wrap,
            width=120,
            height=22,
            bg=self.theme["log_bg"],
            fg=self.theme["log_fg"],
            insertbackground=self.theme["log_fg"],
            font=("Menlo", 10),
            relief=tk.FLAT,
            padx=12,
            pady=10,
            state=tk.DISABLED,
        )
        self.log_scroll = ttk.Scrollbar(log_wrap, orient=tk.VERTICAL, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=self.log_scroll.set)
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)

        footer = tk.Frame(root, bg=self.theme["bg"])
        footer.pack(fill=tk.X, pady=(10, 0))
        tk.Label(footer, text="STATUS", bg=self.theme["highlight"], fg="#0F172A", font=("Avenir Next", 9, "bold"), padx=8, pady=4).pack(side=tk.LEFT)
        tk.Label(footer, textvariable=self.status_var, bg=self.theme["bg"], fg=self.theme["text"], font=("Menlo", 11, "bold")).pack(side=tk.LEFT, padx=(10, 0))
        tk.Label(footer, text="DeepSORT tracking / CLEAR MOT evaluation / Safe result cleanup", bg=self.theme["bg"], fg=self.theme["muted"], font=("Avenir Next", 10)).pack(side=tk.RIGHT)

    def _build_hero(self, parent):
        hero = tk.Frame(parent, bg=self.theme["hero_bg"], highlightbackground=self.theme["hero_border"], highlightthickness=1)
        hero.pack(fill=tk.X, pady=(0, 8))

        left = tk.Frame(hero, bg=self.theme["hero_bg"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=18, pady=14)

        tk.Label(
            left,
            text="MOT CONTROL DECK",
            bg=self.theme["highlight"],
            fg="#0F172A",
            font=("Avenir Next", 10, "bold"),
            padx=10,
            pady=4,
        ).pack(anchor="w")
        tk.Label(
            left,
            text="Multi-Object Tracking Console",
            bg=self.theme["hero_bg"],
            fg=self.theme["hero_text"],
            font=("Avenir Next", 24, "bold"),
        ).pack(anchor="w", pady=(8, 2))
        tk.Label(
            left,
            text="YOLOv5 / YOLOv8 + DeepSORT + CLEAR MOT",
            bg=self.theme["hero_bg"],
            fg=self.theme["hero_muted"],
            font=("Avenir Next", 12),
        ).pack(anchor="w")

        chips = tk.Frame(left, bg=self.theme["hero_bg"])
        chips.pack(anchor="w", pady=(10, 0))
        self._add_chip(chips, self.hero_detector_var, "#123B37")
        self._add_chip(chips, self.hero_results_var, "#4A2D0C")
        self._add_chip(chips, self.hero_workspace_var, "#1E2F4D")

        stats = tk.Frame(hero, bg=self.theme["hero_bg"])
        stats.pack(side=tk.RIGHT, padx=14, pady=12)
        self._build_hero_stat(stats, "Tracking Runs", self.tracking_runs_var, "#173C37")
        self._build_hero_stat(stats, "Eval Runs", self.pipeline_runs_var, "#4A2D0C")
        self._build_hero_stat(stats, "Result Size", self.result_size_var, "#1E2F4D")

    def _add_chip(self, parent, variable: tk.StringVar, bg_color: str):
        tk.Label(
            parent,
            textvariable=variable,
            bg=bg_color,
            fg=self.theme["hero_text"],
            font=("Avenir Next", 9, "bold"),
            padx=10,
            pady=5,
        ).pack(side=tk.LEFT, padx=(0, 8))

    def _build_hero_stat(self, parent, title: str, variable: tk.StringVar, bg_color: str):
        card = tk.Frame(parent, bg=bg_color, width=152, height=64)
        card.pack(fill=tk.X, pady=3)
        card.pack_propagate(False)
        tk.Label(card, text=title, bg=bg_color, fg=self.theme["hero_muted"], font=("Avenir Next", 9)).pack(anchor="w", padx=10, pady=(7, 1))
        tk.Label(card, textvariable=variable, bg=bg_color, fg=self.theme["hero_text"], font=("Avenir Next", 18, "bold")).pack(anchor="w", padx=10)

    def _build_toolbar(self, parent):
        toolbar = tk.Frame(parent, bg=self.theme["bg"])
        toolbar.pack(fill=tk.X, pady=(0, 8))

        self.start_btn = ttk.Button(toolbar, text="Start Tracking", style="Primary.Action.TButton", command=self.start_tracking)
        self.start_btn.pack(side=tk.LEFT)
        self.pipeline_btn = ttk.Button(toolbar, text="Run Evaluation", style="Secondary.Action.TButton", command=self.start_pipeline)
        self.pipeline_btn.pack(side=tk.LEFT, padx=(10, 0))
        self.clear_btn = ttk.Button(toolbar, text="Clear Tracking Results", style="Danger.Action.TButton", command=self.clear_tracking_results)
        self.clear_btn.pack(side=tk.LEFT, padx=(10, 0))
        self.stop_btn = ttk.Button(toolbar, text="Stop Task", style="Ghost.Action.TButton", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(
            toolbar,
            text="Training artifacts are preserved. Cleanup only targets tracking / pipeline / eval outputs.",
            bg=self.theme["bg"],
            fg=self.theme["muted"],
            font=("Avenir Next", 10),
        ).pack(side=tk.RIGHT)

    def _build_results_workspace(self, parent):
        workspace = ttk.LabelFrame(parent, text="Results Workspace", style="Card.TLabelframe")
        workspace.pack(fill=tk.X, pady=(0, 8))

        stats_row = tk.Frame(workspace, bg=self.theme["panel"])
        stats_row.pack(fill=tk.X, padx=10, pady=(10, 6))
        self._build_metric_tile(stats_row, "Tracking Runs", self.tracking_runs_var, "#0F766E")
        self._build_metric_tile(stats_row, "Eval Runs", self.pipeline_runs_var, "#D97706")
        self._build_metric_tile(stats_row, "Result Files", self.result_files_var, "#2563EB")

        info = tk.Frame(workspace, bg=self.theme["panel"])
        info.pack(fill=tk.X, padx=10, pady=(0, 6))
        tk.Label(info, text="Results Directory", bg=self.theme["panel"], fg=self.theme["muted"], font=("Avenir Next", 10, "bold")).pack(anchor="w")
        tk.Label(info, textvariable=self.result_root_var, bg=self.theme["panel"], fg=self.theme["text"], font=("Menlo", 9), justify=tk.LEFT, wraplength=320).pack(anchor="w", pady=(4, 0))

        buttons = tk.Frame(workspace, bg=self.theme["panel"])
        buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.open_results_btn = ttk.Button(buttons, text="Open Results Folder", style="Ghost.Action.TButton", command=self.open_results_root)
        self.open_results_btn.pack(side=tk.LEFT)
        self.refresh_results_btn = ttk.Button(buttons, text="Refresh Stats", style="Ghost.Action.TButton", command=self._refresh_result_summary)
        self.refresh_results_btn.pack(side=tk.LEFT, padx=(10, 0))

    def _build_metric_tile(self, parent, title: str, variable: tk.StringVar, accent: str):
        tile = tk.Frame(parent, bg=self.theme["panel_alt"], highlightbackground=self.theme["border"], highlightthickness=1)
        tile.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        tk.Frame(tile, bg=accent, width=4).pack(side=tk.LEFT, fill=tk.Y)
        body = tk.Frame(tile, bg=self.theme["panel_alt"])
        body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=8)
        tk.Label(body, text=title, bg=self.theme["panel_alt"], fg=self.theme["muted"], font=("Avenir Next", 9)).pack(anchor="w")
        tk.Label(body, textvariable=variable, bg=self.theme["panel_alt"], fg=self.theme["text"], font=("Avenir Next", 18, "bold")).pack(anchor="w")

    def _build_eval_form(self, parent):
        eval_form = ttk.LabelFrame(parent, text="Evaluation Pipeline", style="Card.TLabelframe")
        eval_form.pack(fill=tk.X)
        eval_form.columnconfigure(1, weight=1)
        pad = {"padx": 10, "pady": 6}

        row = 0
        ttk.Label(eval_form, text="Ground truth file (gt.txt)", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.gt_var = tk.StringVar()
        ttk.Entry(eval_form, textvariable=self.gt_var, width=40).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(eval_form, text="Browse File", style="Ghost.Action.TButton", command=self._pick_gt).grid(row=row, column=2, sticky="e", padx=(0, 10), pady=6)

        row += 1
        ttk.Label(eval_form, text="Output root directory", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.run_dir_var = tk.StringVar()
        ttk.Entry(eval_form, textvariable=self.run_dir_var, width=40, state="readonly").grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(eval_form, text="Reset Default", style="Ghost.Action.TButton", command=self._refresh_run_dir).grid(row=row, column=2, sticky="e", padx=(0, 10), pady=6)

        row += 1
        ttk.Label(eval_form, text="Evaluation IoU threshold", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.eval_iou_var = tk.StringVar(value="0.5")
        ttk.Entry(eval_form, textvariable=self.eval_iou_var, width=16).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(eval_form, text="GT class ID", style="Field.TLabel").grid(row=row, column=0, sticky="w")
        self.gt_class_id_var = tk.StringVar(value="1")
        ttk.Entry(eval_form, textvariable=self.gt_class_id_var, width=16).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(
            eval_form,
            text="The pipeline creates a new pipeline_<timestamp> directory under the output root and exports video, MOT text, JSON, CSV, and SVG charts automatically.",
            style="Hint.TLabel",
            wraplength=360,
            justify=tk.LEFT,
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))

    def _bind_dynamic_state(self):
        self.yolo_version_var.trace_add("write", lambda *_: self._refresh_result_summary())
        self.yolo_var.trace_add("write", lambda *_: self._refresh_result_summary())
        self.run_dir_var.trace_add("write", lambda *_: self._refresh_result_summary())

    def _fill_defaults(self):
        self.source_var.set("0")
        self.source_preset_var.set("Webcam (0)")
        self.yolo_version_var.set("yolov5")
        self.yolo_var.set(str(ROOT / "yolov5s.pt"))
        self.reid_var.set(str(ROOT / "osnet_x0_25_msmt17.pt"))
        self.imgsz_var.set("640")
        self.max_det_var.set("150")
        self.person_only_var.set(True)
        self.run_dir_var.set(str(ROOT / "runs"))

    def _set_status(self, status: str):
        self.status_var.set(status)

    def _refresh_run_dir(self):
        self.run_dir_var.set(str(ROOT / "runs"))

    def _prepare_run_dir(self, prefix: str) -> Path:
        base_dir = self._get_results_root()
        base_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"{prefix}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir_var.set(str(base_dir))
        return run_dir

    def _get_results_root(self) -> Path:
        raw = self.run_dir_var.get().strip()
        return Path(raw) if raw else ROOT / "runs"

    def _on_yolo_version_change(self, _event=None):
        version = self.yolo_version_var.get().strip()
        current_raw = self.yolo_var.get().strip()
        current = Path(current_raw) if current_raw else None
        inferred = self._infer_yolo_version_from_weights(current_raw)
        if current and current.exists() and inferred in (None, version):
            return
        if version == "yolov8":
            best = ROOT / "runs" / "train" / "yolov8_mot16" / "weights" / "best.pt"
            self.yolo_var.set(str(best if best.exists() else ROOT / "yolov8n.pt"))
        else:
            self.yolo_var.set(str(ROOT / "yolov5s.pt"))

    @staticmethod
    def _infer_yolo_version_from_weights(weights_path: str) -> str | None:
        normalized = str(weights_path).lower()
        if "yolov8" in normalized:
            return "yolov8"
        if "yolov5" in normalized:
            return "yolov5"
        return None

    def _pick_source(self):
        path = filedialog.askopenfilename(title="Select video source", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")])
        if path:
            self.source_var.set(path)
            self.source_preset_var.set("Custom")

    def _on_source_preset_change(self, _event=None):
        preset = self.source_preset_var.get()
        if preset == "Webcam (0)":
            self.source_var.set("0")
        elif preset == "Demo video 1 (test_video.mp4)":
            self.source_var.set(str(ROOT / "test_video.mp4"))
        elif preset == "Demo video 2 (test_video_2.mp4)":
            self.source_var.set(str(ROOT / "test_video_2.mp4"))

    def _pick_file(self, target_var: tk.StringVar):
        path = filedialog.askopenfilename(title="Select weight file", filetypes=[("Model Files", "*.pt"), ("All Files", "*.*")])
        if path:
            target_var.set(path)

    def _pick_yolo_file(self):
        path = filedialog.askopenfilename(title="Select YOLO weight file", filetypes=[("Model Files", "*.pt"), ("All Files", "*.*")])
        if path:
            self.yolo_var.set(path)
            inferred = self._infer_yolo_version_from_weights(path)
            if inferred is not None:
                self.yolo_version_var.set(inferred)

    def _pick_gt(self):
        path = filedialog.askopenfilename(title="Select ground-truth file", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if path:
            self.gt_var.set(path)

    def _append_log(self, text: str):
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _validate_inputs(self):
        source = self.source_var.get().strip()
        if not source:
            raise ValueError("Source cannot be empty.")
        if not Path(self.yolo_var.get().strip()).exists():
            raise ValueError(f"YOLO weight not found: {self.yolo_var.get().strip()}")
        if not Path(self.reid_var.get().strip()).exists():
            raise ValueError(f"ReID weight not found: {self.reid_var.get().strip()}")
        inferred = self._infer_yolo_version_from_weights(self.yolo_var.get().strip())
        if inferred is not None and inferred != self.yolo_version_var.get().strip():
            self.yolo_version_var.set(inferred)
        conf = float(self.conf_var.get().strip())
        iou = float(self.iou_var.get().strip())
        imgsz = int(self.imgsz_var.get().strip())
        max_det = int(self.max_det_var.get().strip())
        if not (0.0 <= conf <= 1.0):
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if not (0.0 <= iou <= 1.0):
            raise ValueError("IoU threshold must be between 0 and 1.")
        if imgsz <= 0:
            raise ValueError("imgsz must be a positive integer.")
        if max_det <= 0:
            raise ValueError("max_det must be a positive integer.")

    def _validate_pipeline_inputs(self):
        self._validate_inputs()
        gt_raw = self.gt_var.get().strip()
        if not gt_raw:
            raise ValueError("Please select a GT annotation file (gt.txt).")
        gt = Path(gt_raw)
        if not gt.exists():
            raise ValueError(f"GT file not found: {gt}")
        if not gt.is_file():
            raise ValueError(f"GT path is not a file: {gt}")
        if gt.suffix.lower() != ".txt":
            raise ValueError("GT file must use the .txt extension.")

    def _list_tracking_artifacts(self):
        base = self._get_results_root()
        artifacts: list[Path] = []

        if base.exists():
            for pattern in ("tracking_*", "pipeline_*"):
                artifacts.extend(sorted(p for p in base.glob(pattern) if p.exists()))
            for legacy_name in ("track", "eval"):
                legacy_path = base / legacy_name
                if legacy_path.exists():
                    artifacts.append(legacy_path)

        artifacts.extend(sorted(ROOT.glob("output*.mp4")))

        unique = {}
        for path in artifacts:
            unique[str(path)] = path
        return list(unique.values())

    def _measure_artifact(self, path: Path) -> tuple[int, int]:
        file_count = 0
        total_size = 0
        try:
            if path.is_file():
                return 1, path.stat().st_size
            if path.is_dir():
                for item in path.rglob("*"):
                    if item.is_file():
                        file_count += 1
                        total_size += item.stat().st_size
        except OSError:
            return file_count, total_size
        return file_count, total_size

    def _format_bytes(self, size: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(size)
        for unit in units:
            if value < 1024.0 or unit == units[-1]:
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024.0
        return f"{size} B"

    def _shorten_path(self, value: str, limit: int = 40) -> str:
        if len(value) <= limit:
            return value
        return f"...{value[-(limit - 3):]}"

    def _refresh_result_summary(self):
        base = self._get_results_root()
        tracking_runs = len(list(base.glob("tracking_*"))) if base.exists() else 0
        pipeline_runs = len(list(base.glob("pipeline_*"))) if base.exists() else 0
        artifacts = self._list_tracking_artifacts()
        total_files = 0
        total_size = 0
        for path in artifacts:
            files, size = self._measure_artifact(path)
            total_files += files
            total_size += size

        detector = self.yolo_version_var.get().strip().upper() if hasattr(self, "yolo_version_var") else "YOLOV5"
        self.hero_detector_var.set(f"{detector} / DEEPSORT")
        self.hero_results_var.set(f"{len(artifacts)} result targets")
        self.hero_workspace_var.set(self._shorten_path(str(base)))
        self.tracking_runs_var.set(str(tracking_runs))
        self.pipeline_runs_var.set(str(pipeline_runs))
        self.result_files_var.set(str(total_files))
        self.result_size_var.set(self._format_bytes(total_size))
        self.result_root_var.set(str(base))

    def _launch_process(self, cmd, task_name: str):
        if self.process is not None:
            return
        self.current_task = task_name
        self._append_log(f"$ {' '.join(cmd)}\n")
        self._append_log(f"{task_name} started.\n")
        self._set_status(task_name)
        self.process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.start_btn.configure(state=tk.DISABLED)
        self.pipeline_btn.configure(state=tk.DISABLED)
        self.clear_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        threading.Thread(target=self._read_process_output, daemon=True).start()
        self.master.after(100, self._poll_process_output)

    def _open_file(self, path: Path):
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            self._append_log(f"Auto-open failed: {path}. Reason: {e}\n")

    def open_results_root(self):
        base = self._get_results_root()
        base.mkdir(parents=True, exist_ok=True)
        self._open_file(base)

    def clear_tracking_results(self):
        """Delete tracking and evaluation outputs without touching training runs."""
        if self.process is not None:
            messagebox.showwarning("Task Running", "Stop the current task before clearing results.")
            return

        artifacts = self._list_tracking_artifacts()
        if not artifacts:
            messagebox.showinfo("Nothing to Clear", "No tracking results are available for cleanup.")
            return

        dir_count = sum(1 for path in artifacts if path.is_dir())
        file_count = sum(1 for path in artifacts if path.is_file())
        preview = "\n".join(f"- {path.name}" for path in artifacts[:6])
        if len(artifacts) > 6:
            preview += f"\n- ... and {len(artifacts) - 6} more"

        confirmed = messagebox.askyesno(
            "Confirm Result Cleanup",
            "This action deletes tracking and evaluation artifacts under the current results directory, but keeps training outputs intact.\n\n"
            f"Directories: {dir_count}\nFiles: {file_count}\n\n"
            f"{preview}",
        )
        if not confirmed:
            return

        deleted = 0
        failures = []
        for path in artifacts:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                deleted += 1
            except Exception as exc:
                failures.append((path, exc))

        self._refresh_result_summary()
        self._append_log(f"Result cleanup completed. Removed {deleted} item(s).\n")
        if failures:
            detail = "\n".join(f"{path}: {exc}" for path, exc in failures[:5])
            messagebox.showwarning("Partial Cleanup Failure", f"{len(failures)} item(s) could not be removed:\n{detail}")
        else:
            messagebox.showinfo("Cleanup Complete", f"Removed {deleted} tracking result item(s).")

    def start_tracking(self):
        if self.process is not None:
            return
        try:
            self._validate_inputs()
        except Exception as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return
        run_dir = self._prepare_run_dir("tracking")
        cmd = [
            self.python_exec,
            str(ROOT / "examples" / "track.py"),
            "--yolo-version",
            self.yolo_version_var.get().strip(),
            "--source",
            self.source_var.get().strip(),
            "--yolo-weights",
            self.yolo_var.get().strip(),
            "--reid-weights",
            self.reid_var.get().strip(),
            "--conf-thres",
            self.conf_var.get().strip(),
            "--iou-thres",
            self.iou_var.get().strip(),
            "--imgsz",
            self.imgsz_var.get().strip(),
            "--max-det",
            self.max_det_var.get().strip(),
            "--output",
            str(run_dir / "tracked.mp4"),
            "--save-mot-txt",
            str(run_dir / "track_mot.txt"),
            "--runtime-json",
            str(run_dir / "runtime.json"),
        ]
        if self.person_only_var.get():
            cmd.append("--person-only")
        else:
            cmd.append("--all-classes")
        if self.show_var.get():
            cmd.append("--show")
        self._launch_process(cmd, "Tracking")

    def start_pipeline(self):
        if self.process is not None:
            return
        try:
            self._validate_pipeline_inputs()
        except Exception as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return

        run_dir = self._prepare_run_dir("pipeline")
        gt_path = Path(self.gt_var.get().strip()).resolve()
        cmd = [
            self.python_exec,
            str(ROOT / "examples" / "run_clear_mot_pipeline.py"),
            "--source",
            self.source_var.get().strip(),
            "--gt",
            str(gt_path),
            "--yolo-version",
            self.yolo_version_var.get().strip(),
            "--yolo-weights",
            self.yolo_var.get().strip(),
            "--reid-weights",
            self.reid_var.get().strip(),
            "--conf-thres",
            self.conf_var.get().strip(),
            "--iou-thres",
            self.iou_var.get().strip(),
            "--imgsz",
            self.imgsz_var.get().strip(),
            "--max-det",
            self.max_det_var.get().strip(),
            "--eval-iou-thr",
            self.eval_iou_var.get().strip(),
            "--gt-class-id",
            self.gt_class_id_var.get().strip(),
            "--run-dir",
            str(run_dir),
        ]
        if self.person_only_var.get():
            cmd.append("--person-only")
        else:
            cmd.append("--all-classes")
        if self.show_var.get():
            cmd.append("--show")
        self._launch_process(cmd, "Evaluation Pipeline")

    def _read_process_output(self):
        if self.process is None or self.process.stdout is None:
            return
        for line in iter(self.process.stdout.readline, ""):
            self.log_queue.put(line)

    def _poll_process_output(self):
        if self.process is None:
            return
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)

        if self.process.poll() is None:
            self.master.after(100, self._poll_process_output)
            return

        rc = self.process.returncode
        task_name = self.current_task or "Task"
        self._append_log(f"\n{task_name} finished (exit code={rc}).\n")
        if rc == 0 and task_name == "Evaluation Pipeline":
            base = self._get_results_root()
            latest = sorted(base.glob("pipeline_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if latest:
                eval_dir = latest[0] / "eval"
                bar_svg = eval_dir / "metrics_bar.svg"
                curve_svg = eval_dir / "mota_curve.svg"
                self._append_log(f"Charts generated:\n- {bar_svg}\n- {curve_svg}\n")
                if bar_svg.exists():
                    self._open_file(bar_svg)
                if curve_svg.exists():
                    self._open_file(curve_svg)

        self.process = None
        self.current_task = ""
        self.start_btn.configure(state=tk.NORMAL)
        self.pipeline_btn.configure(state=tk.NORMAL)
        self.clear_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._set_status("Idle")
        self._refresh_result_summary()

    def stop_tracking(self):
        if self.process is None:
            return
        self._append_log("Stopping task...\n")
        self.process.terminate()
        self._set_status("Stopping...")

    def on_close(self):
        if self.process is not None:
            self.process.terminate()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = TrackingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
