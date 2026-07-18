import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from sign_detector import (
    ALL_ACTIONS,
    DATA_PATH,
    SignLanguageDetector,
    create_capture_session,
    dataset_sample_counts,
    load_fast_classifier,
    save_training_sample,
)


COLORS = {
    "background": "#07111F",
    "surface": "#0E1B2E",
    "surface_alt": "#13233A",
    "border": "#20334F",
    "text": "#F4F7FB",
    "muted": "#91A4BE",
    "primary": "#37CEA3",
    "accent": "#70A5FF",
    "warning": "#FFBE55",
    "danger": "#FF6B7A",
}

CAPTURE_TARGET = 60
CAPTURE_COUNTDOWN_SECONDS = 2
CAPTURE_EVERY_N_FRAMES = 2


def open_camera():
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    for backend in backends:
        for index in range(3):
            camera = cv2.VideoCapture(index, backend)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, _frame = camera.read()
                if ok:
                    return camera, index
            camera.release()
    return None, None


class SignLanguageApp:
    def __init__(self, root, initial_mode="detect"):
        self.root = root
        self.root.title("SignFlow | Sign Language Studio")
        self.root.geometry("1320x850")
        self.root.minsize(1120, 720)
        self.root.configure(bg=COLORS["background"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("<Control-l>", lambda _event: self.clear_sentence())

        self.detector = None
        self.camera = None
        self.camera_index = None
        self.running = True
        self.mode = "detect"
        self.initial_mode = initial_mode
        self.photo = None
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0

        self.capturing = False
        self.capture_session = None
        self.capture_count = 0
        self.capture_frame_counter = 0
        self.capture_started_at = 0.0

        self._build_ui()
        self._set_loading_state("Loading recognition engine...")
        self.root.after(80, self._initialize)

    def _build_ui(self):
        header = tk.Frame(self.root, bg=COLORS["background"], height=100)
        header.pack(fill="x", padx=34, pady=(20, 10))
        header.pack_propagate(False)

        brand = tk.Frame(header, bg=COLORS["background"])
        brand.pack(side="left", fill="y")
        tk.Label(
            brand,
            text="SignFlow",
            font=("Segoe UI Semibold", 25),
            fg=COLORS["text"],
            bg=COLORS["background"],
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="SIGN LANGUAGE TRAINING AND DETECTION STUDIO",
            font=("Segoe UI Semibold", 9),
            fg=COLORS["primary"],
            bg=COLORS["background"],
        ).pack(anchor="w", pady=(2, 0))

        mode_switch = tk.Frame(header, bg=COLORS["surface_alt"])
        mode_switch.pack(side="left", padx=(70, 0), pady=12)
        self.detect_mode_button = self._button(
            mode_switch,
            "Detection",
            lambda: self.switch_mode("detect"),
            COLORS["primary"],
            "#04130F",
        )
        self.detect_mode_button.pack(side="left", padx=4, pady=4)
        self.train_mode_button = self._button(
            mode_switch,
            "Training",
            lambda: self.switch_mode("train"),
            COLORS["surface_alt"],
            COLORS["muted"],
        )
        self.train_mode_button.pack(side="left", padx=4, pady=4)

        self.live_badge = tk.Label(
            header,
            text="  INITIALIZING  ",
            font=("Segoe UI Semibold", 9),
            fg=COLORS["warning"],
            bg=COLORS["surface_alt"],
            padx=12,
            pady=8,
        )
        self.live_badge.pack(side="right", pady=14)

        content = tk.Frame(self.root, bg=COLORS["background"])
        content.pack(fill="both", expand=True, padx=34, pady=(0, 24))
        content.grid_columnconfigure(0, weight=3, uniform="content")
        content.grid_columnconfigure(1, weight=2, uniform="content")
        content.grid_rowconfigure(0, weight=1)

        self._build_camera_panel(content)

        self.workspace = tk.Frame(content, bg=COLORS["background"])
        self.workspace.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(0, weight=1)

        self.detection_panel = tk.Frame(self.workspace, bg=COLORS["background"])
        self.training_panel = tk.Frame(self.workspace, bg=COLORS["background"])
        for panel in (self.detection_panel, self.training_panel):
            panel.grid(row=0, column=0, sticky="nsew")
            panel.grid_columnconfigure(0, weight=1)

        self._build_detection_panel()
        self._build_training_panel()
        self.detection_panel.tkraise()

    def _build_camera_panel(self, parent):
        left = self._card(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        camera_header = tk.Frame(left, bg=COLORS["surface"])
        camera_header.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 12))
        self.camera_title = tk.Label(
            camera_header,
            text="Live detection camera",
            font=("Segoe UI Semibold", 15),
            fg=COLORS["text"],
            bg=COLORS["surface"],
        )
        self.camera_title.pack(side="left")
        self.camera_meta = tk.Label(
            camera_header,
            text="Preparing camera",
            font=("Segoe UI", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        )
        self.camera_meta.pack(side="right")

        shell = tk.Frame(
            left,
            bg="#030810",
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        shell.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 16))
        shell.grid_rowconfigure(0, weight=1)
        shell.grid_columnconfigure(0, weight=1)
        self.video_label = tk.Label(
            shell,
            text="Starting camera...",
            font=("Segoe UI", 13),
            fg=COLORS["muted"],
            bg="#030810",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        footer = tk.Frame(left, bg=COLORS["surface"])
        footer.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 18))
        self.hand_status = tk.Label(
            footer,
            text="*  Waiting for hand",
            font=("Segoe UI Semibold", 10),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        )
        self.hand_status.pack(side="left")
        self.camera_hint = tk.Label(
            footer,
            text="Keep one hand inside the guide",
            font=("Segoe UI", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        )
        self.camera_hint.pack(side="right")

    def _build_detection_panel(self):
        self.detection_panel.grid_rowconfigure(1, weight=1)

        result = self._card(self.detection_panel)
        result.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        title = tk.Frame(result, bg=COLORS["surface"])
        title.pack(fill="x", padx=24, pady=(20, 0))
        tk.Label(
            title,
            text="CURRENT PREDICTION",
            font=("Segoe UI Semibold", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(side="left")
        self.fps_label = tk.Label(
            title,
            text="0 FPS",
            font=("Segoe UI", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        )
        self.fps_label.pack(side="right")

        row = tk.Frame(result, bg=COLORS["surface"])
        row.pack(fill="x", padx=24, pady=(7, 12))
        self.sign_label = tk.Label(
            row,
            text="?",
            width=2,
            anchor="w",
            font=("Segoe UI Semibold", 66),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        )
        self.sign_label.pack(side="left")
        details = tk.Frame(row, bg=COLORS["surface"])
        details.pack(side="left", fill="x", expand=True, padx=(16, 0))
        self.prediction_status = tk.Label(
            details,
            text="Waiting for a stable sign",
            font=("Segoe UI Semibold", 12),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.prediction_status.pack(fill="x", pady=(12, 9))
        self.confidence_canvas = tk.Canvas(
            details,
            height=9,
            bg=COLORS["surface_alt"],
            highlightthickness=0,
        )
        self.confidence_canvas.pack(fill="x")
        self.confidence_text = tk.Label(
            details,
            text="Confidence 0%",
            font=("Segoe UI", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.confidence_text.pack(fill="x", pady=(7, 0))

        sequence = self._card(self.detection_panel)
        sequence.grid(row=1, column=0, sticky="nsew", pady=(0, 14))
        sequence.grid_columnconfigure(0, weight=1)
        sequence.grid_rowconfigure(1, weight=1)
        heading = tk.Frame(sequence, bg=COLORS["surface"])
        heading.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 10))
        tk.Label(
            heading,
            text="Detected sequence",
            font=("Segoe UI Semibold", 15),
            fg=COLORS["text"],
            bg=COLORS["surface"],
        ).pack(side="left")
        tk.Label(
            heading,
            text="Ctrl + L to clear",
            font=("Segoe UI", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(side="right")
        self.sentence_label = tk.Label(
            sequence,
            text="Your detected signs will appear here.",
            font=("Segoe UI Semibold", 20),
            fg=COLORS["muted"],
            bg=COLORS["surface_alt"],
            anchor="nw",
            justify="left",
            wraplength=420,
            padx=18,
            pady=18,
        )
        self.sentence_label.grid(
            row=1, column=0, sticky="nsew", padx=24, pady=(0, 14)
        )
        self.supported_frame = tk.Frame(sequence, bg=COLORS["surface"])
        self.supported_frame.grid(
            row=2, column=0, sticky="ew", padx=24, pady=(0, 18)
        )

        actions = self._card(self.detection_panel)
        actions.grid(row=2, column=0, sticky="ew")
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        self._button(
            actions,
            "Clear sequence",
            self.clear_sentence,
            COLORS["surface_alt"],
            COLORS["text"],
        ).grid(row=0, column=0, sticky="ew", padx=(20, 7), pady=18)
        self._button(
            actions,
            "Exit application",
            self.close,
            COLORS["primary"],
            "#04130F",
        ).grid(row=0, column=1, sticky="ew", padx=(7, 20), pady=18)

    def _build_training_panel(self):
        self.training_panel.grid_rowconfigure(1, weight=1)

        setup = self._card(self.training_panel)
        setup.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        tk.Label(
            setup,
            text="Capture a sign",
            font=("Segoe UI Semibold", 17),
            fg=COLORS["text"],
            bg=COLORS["surface"],
        ).pack(anchor="w", padx=24, pady=(20, 4))
        tk.Label(
            setup,
            text="Choose a letter, hold the pose naturally, and vary the angle slightly.",
            font=("Segoe UI", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(anchor="w", padx=24)

        form = tk.Frame(setup, bg=COLORS["surface"])
        form.pack(fill="x", padx=24, pady=18)
        tk.Label(
            form,
            text="SIGN LABEL",
            font=("Segoe UI Semibold", 8),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(anchor="w")
        self.training_label = tk.StringVar(value="A")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "SignFlow.TCombobox",
            fieldbackground=COLORS["surface_alt"],
            background=COLORS["surface_alt"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            arrowsize=16,
            padding=9,
        )
        self.label_picker = ttk.Combobox(
            form,
            textvariable=self.training_label,
            values=ALL_ACTIONS,
            state="readonly",
            style="SignFlow.TCombobox",
            font=("Segoe UI Semibold", 11),
        )
        self.label_picker.pack(fill="x", pady=(6, 0))
        self.label_picker.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_dataset_summary(),
        )

        capture = self._card(self.training_panel)
        capture.grid(row=1, column=0, sticky="nsew", pady=(0, 14))
        capture.grid_columnconfigure(0, weight=1)
        capture.grid_rowconfigure(2, weight=1)
        tk.Label(
            capture,
            text="Training session",
            font=("Segoe UI Semibold", 15),
            fg=COLORS["text"],
            bg=COLORS["surface"],
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(20, 4))
        self.training_status = tk.Label(
            capture,
            text="Ready to capture 60 landmark samples.",
            font=("Segoe UI", 10),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            anchor="w",
        )
        self.training_status.grid(row=1, column=0, sticky="ew", padx=24)

        progress_area = tk.Frame(capture, bg=COLORS["surface_alt"])
        progress_area.grid(
            row=2, column=0, sticky="nsew", padx=24, pady=(16, 14)
        )
        self.capture_number = tk.Label(
            progress_area,
            text="0 / 60",
            font=("Segoe UI Semibold", 36),
            fg=COLORS["text"],
            bg=COLORS["surface_alt"],
        )
        self.capture_number.pack(pady=(26, 8))
        self.capture_progress = tk.Canvas(
            progress_area,
            height=10,
            bg=COLORS["border"],
            highlightthickness=0,
        )
        self.capture_progress.pack(fill="x", padx=28)
        self.dataset_summary = tk.Label(
            progress_area,
            text="Loading dataset statistics...",
            font=("Segoe UI", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface_alt"],
        )
        self.dataset_summary.pack(pady=(12, 24))

        controls = tk.Frame(capture, bg=COLORS["surface"])
        controls.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 20))
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)
        self.capture_button = self._button(
            controls,
            "Start capture",
            self.start_capture,
            COLORS["primary"],
            "#04130F",
        )
        self.capture_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 7)
        )
        self.rebuild_button = self._button(
            controls,
            "Rebuild model",
            self.rebuild_classifier,
            COLORS["surface_alt"],
            COLORS["text"],
        )
        self.rebuild_button.grid(
            row=0, column=1, sticky="ew", padx=(7, 0)
        )

        notes = self._card(self.training_panel)
        notes.grid(row=2, column=0, sticky="ew")
        tk.Label(
            notes,
            text=(
                "Training tips\n"
                "Use a plain background and good light. Keep the full hand visible. "
                "During capture, slowly vary distance and angle without changing the sign."
            ),
            font=("Segoe UI", 9),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
            justify="left",
            wraplength=430,
            padx=20,
            pady=16,
        ).pack(fill="x")

    @staticmethod
    def _card(parent):
        return tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )

    @staticmethod
    def _button(parent, text, command, background, foreground):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI Semibold", 10),
            fg=foreground,
            bg=background,
            activeforeground=foreground,
            activebackground=background,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=16,
            pady=11,
        )

    def _initialize(self):
        try:
            self.detector = SignLanguageDetector(
                classifier=load_fast_classifier()
            )
            self.camera, self.camera_index = open_camera()
            if self.camera is None:
                raise RuntimeError(
                    "No camera is available. Close other camera apps and try again."
                )
            self.live_badge.configure(
                text="  *  LIVE  ", fg=COLORS["primary"], bg="#102E2A"
            )
            self.camera_meta.configure(
                text=f"Camera {self.camera_index}  |  640 x 480"
            )
            self._refresh_supported_labels()
            self._refresh_dataset_summary()
            if self.initial_mode == "train":
                self.switch_mode("train")
            self._update_frame()
        except Exception as exc:
            self._show_error(str(exc))

    def switch_mode(self, mode):
        if mode == self.mode:
            return
        if self.capturing and mode == "detect":
            self.stop_capture()
        self.mode = mode
        if mode == "detect":
            self.detection_panel.tkraise()
            self.camera_title.configure(text="Live detection camera")
            self.camera_hint.configure(text="Keep one hand inside the guide")
            self.detect_mode_button.configure(
                bg=COLORS["primary"], fg="#04130F"
            )
            self.train_mode_button.configure(
                bg=COLORS["surface_alt"], fg=COLORS["muted"]
            )
            self.detector.reset()
        else:
            self.training_panel.tkraise()
            self.camera_title.configure(text="Training capture camera")
            self.camera_hint.configure(text="Hold the selected sign inside the guide")
            self.detect_mode_button.configure(
                bg=COLORS["surface_alt"], fg=COLORS["muted"]
            )
            self.train_mode_button.configure(
                bg=COLORS["primary"], fg="#04130F"
            )
            self.detector.reset()
            self._refresh_dataset_summary()

    def _update_frame(self):
        if not self.running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self._show_error("The camera stopped returning frames.")
            return

        try:
            frame = cv2.flip(frame, 1)
            processed = self.detector.process(
                frame,
                draw_status=False,
                classify=self.mode == "detect",
            )
            state = self.detector.get_state()
            if self.mode == "detect":
                self._update_detection_metrics(state)
            else:
                self._update_training_capture(state)
            self._update_hand_status(state["hand_visible"])
            self._render_video(processed)
        except Exception as exc:
            self._show_error(f"Processing error: {exc}")
            return

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.last_frame_time, 0.001)
        self.fps = instant_fps if self.fps == 0 else self.fps * 0.88 + instant_fps * 0.12
        self.last_frame_time = now
        self.fps_label.configure(text=f"{self.fps:.0f} FPS")
        self.root.after(10, self._update_frame)

    def _update_training_capture(self, state):
        if not self.capturing:
            return
        remaining = self.capture_started_at - time.perf_counter()
        if remaining > 0:
            self.training_status.configure(
                text=f"Get ready... capture begins in {remaining:.1f}s",
                fg=COLORS["warning"],
            )
            return
        if not state["hand_visible"] or state["keypoints"] is None:
            self.training_status.configure(
                text="Show the full hand inside the guide to continue.",
                fg=COLORS["warning"],
            )
            return

        self.capture_frame_counter += 1
        if self.capture_frame_counter % CAPTURE_EVERY_N_FRAMES:
            return
        save_training_sample(
            self.capture_session,
            self.capture_count,
            state["keypoints"],
        )
        self.capture_count += 1
        self.training_status.configure(
            text="Capturing... slowly vary hand angle and distance.",
            fg=COLORS["primary"],
        )
        self._update_capture_progress()
        if self.capture_count >= CAPTURE_TARGET:
            self.finish_capture()

    def start_capture(self):
        if self.capturing:
            self.stop_capture()
            return
        label = self.training_label.get()
        try:
            self.capture_session = create_capture_session(label)
        except Exception as exc:
            messagebox.showerror("Training Mode", str(exc))
            return
        self.capturing = True
        self.capture_count = 0
        self.capture_frame_counter = 0
        self.capture_started_at = (
            time.perf_counter() + CAPTURE_COUNTDOWN_SECONDS
        )
        self.capture_button.configure(
            text="Stop capture",
            bg=COLORS["danger"],
            fg=COLORS["text"],
        )
        self.label_picker.configure(state="disabled")
        self.training_status.configure(
            text="Get ready...", fg=COLORS["warning"]
        )
        self._update_capture_progress()

    def stop_capture(self):
        if not self.capturing:
            return
        self.capturing = False
        self.capture_button.configure(
            text="Start capture",
            bg=COLORS["primary"],
            fg="#04130F",
        )
        self.label_picker.configure(state="readonly")
        if self.capture_count:
            self.training_status.configure(
                text=f"Stopped with {self.capture_count} saved samples.",
                fg=COLORS["warning"],
            )
            self.rebuild_classifier()
        else:
            self.training_status.configure(
                text="Capture stopped. No samples were saved.",
                fg=COLORS["muted"],
            )

    def finish_capture(self):
        self.capturing = False
        self.capture_button.configure(
            text="Start capture",
            bg=COLORS["primary"],
            fg="#04130F",
        )
        self.label_picker.configure(state="readonly")
        self.training_status.configure(
            text="Capture complete. Updating recognition model...",
            fg=COLORS["accent"],
        )
        self.root.update_idletasks()
        self.rebuild_classifier(show_message=False)
        label = self.training_label.get()
        self.training_status.configure(
            text=f"Sign {label} is trained and ready for detection.",
            fg=COLORS["primary"],
        )
        messagebox.showinfo(
            "Training Complete",
            f"Captured {self.capture_count} samples for sign {label}.\n"
            "The detection model has been updated.",
        )

    def rebuild_classifier(self, show_message=True):
        try:
            self.rebuild_button.configure(state="disabled")
            self.training_status.configure(
                text="Rebuilding recognition model...",
                fg=COLORS["accent"],
            )
            self.root.update_idletasks()
            classifier = load_fast_classifier(force_rebuild=True)
            self.detector.set_classifier(classifier)
            self._refresh_supported_labels()
            self._refresh_dataset_summary()
            if show_message:
                self.training_status.configure(
                    text="Recognition model is up to date.",
                    fg=COLORS["primary"],
                )
                messagebox.showinfo(
                    "Model Updated",
                    "Training data was rebuilt successfully.",
                )
        except Exception as exc:
            messagebox.showerror("Training Mode", str(exc))
        finally:
            self.rebuild_button.configure(state="normal")

    def _refresh_supported_labels(self):
        for widget in self.supported_frame.winfo_children():
            widget.destroy()
        tk.Label(
            self.supported_frame,
            text="SUPPORTED",
            font=("Segoe UI Semibold", 8),
            fg=COLORS["muted"],
            bg=COLORS["surface"],
        ).pack(side="left", padx=(0, 10))
        for action in self.detector.actions:
            tk.Label(
                self.supported_frame,
                text=str(action),
                font=("Segoe UI Semibold", 9),
                fg=COLORS["accent"],
                bg=COLORS["surface_alt"],
                padx=8,
                pady=5,
            ).pack(side="left", padx=2)

    def _refresh_dataset_summary(self):
        counts = dataset_sample_counts()
        label = self.training_label.get()
        selected_count = counts.get(label, 0)
        self.dataset_summary.configure(
            text=(
                f"{len(counts)} trained signs  |  "
                f"{sum(counts.values()):,} total samples  |  "
                f"{label}: {selected_count:,}"
            )
        )

    def _update_capture_progress(self):
        self.capture_number.configure(
            text=f"{self.capture_count} / {CAPTURE_TARGET}"
        )
        self.capture_progress.delete("all")
        width = max(self.capture_progress.winfo_width(), 1)
        progress = min(self.capture_count / CAPTURE_TARGET, 1.0)
        self.capture_progress.create_rectangle(
            0, 0, width, 10, fill=COLORS["border"], outline=""
        )
        self.capture_progress.create_rectangle(
            0, 0, width * progress, 10, fill=COLORS["primary"], outline=""
        )

    def _render_video(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width = max(self.video_label.winfo_width() - 8, 320)
        height = max(self.video_label.winfo_height() - 8, 240)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), "#030810")
        canvas.paste(
            image,
            ((width - image.width) // 2, (height - image.height) // 2),
        )
        self.photo = ImageTk.PhotoImage(canvas)
        self.video_label.configure(image=self.photo, text="")

    def _update_detection_metrics(self, state):
        label = state["label"]
        candidate = state["candidate_label"]
        confidence = (
            state["confidence"] if label else state["candidate_confidence"]
        )
        self.sign_label.configure(
            text=label or candidate or "?",
            fg=(
                COLORS["text"]
                if label
                else COLORS["accent"]
                if candidate
                else COLORS["muted"]
            ),
        )
        self.confidence_text.configure(text=f"Confidence {confidence:.0%}")
        self._draw_confidence(confidence)
        if label:
            status, color = "Stable sign detected", COLORS["primary"]
        elif state["hand_visible"]:
            progress = min(
                state["frames_ready"] / state["warmup_frames"], 1.0
            )
            status = (
                f"Checking {candidate}..."
                if candidate
                else f"Reading sign  {progress:.0%}"
            )
            color = COLORS["accent"]
        else:
            status, color = "Waiting for a stable sign", COLORS["muted"]
        self.prediction_status.configure(text=status, fg=color)
        sentence = "  ".join(state["sentence"])
        self.sentence_label.configure(
            text=sentence or "Your detected signs will appear here.",
            fg=COLORS["text"] if sentence else COLORS["muted"],
        )

    def _update_hand_status(self, visible):
        self.hand_status.configure(
            text="*  Hand detected" if visible else "*  Waiting for hand",
            fg=COLORS["primary"] if visible else COLORS["muted"],
        )

    def _draw_confidence(self, confidence):
        self.confidence_canvas.delete("all")
        width = max(self.confidence_canvas.winfo_width(), 1)
        self.confidence_canvas.create_rectangle(
            0, 0, width, 9, fill=COLORS["surface_alt"], outline=""
        )
        self.confidence_canvas.create_rectangle(
            0, 0, width * confidence, 9, fill=COLORS["primary"], outline=""
        )

    def _set_loading_state(self, message):
        self.video_label.configure(text=message, image="")

    def _show_error(self, message):
        self.running = False
        self.live_badge.configure(
            text="  *  OFFLINE  ", fg=COLORS["danger"], bg="#351923"
        )
        self.video_label.configure(
            image="",
            text=f"Camera unavailable\n\n{message}",
            fg=COLORS["danger"],
        )
        messagebox.showerror("SignFlow", message)

    def clear_sentence(self):
        if self.detector:
            self.detector.reset()
        self.sign_label.configure(text="?", fg=COLORS["muted"])
        self.sentence_label.configure(
            text="Your detected signs will appear here.",
            fg=COLORS["muted"],
        )
        self.prediction_status.configure(
            text="Waiting for a stable sign", fg=COLORS["muted"]
        )
        self.confidence_text.configure(text="Confidence 0%")
        self._draw_confidence(0)

    def close(self):
        self.running = False
        if self.camera:
            self.camera.release()
        if self.detector:
            self.detector.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    initial_mode = "train" if "--train" in sys.argv else "detect"
    SignLanguageApp(root, initial_mode=initial_mode)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
