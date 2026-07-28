from __future__ import annotations

import argparse
import base64
from pathlib import Path
import shutil
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import yaml

from auto_label import (
    build_model_specs,
    generate_auto_label_file,
    load_models,
    normalize_threshold,
)
from review_labels import (
    BBox,
    BBoxHistory,
    bbox_from_canvas_rect,
    bbox_to_canvas_rect,
    change_bbox_class,
    create_bbox,
    delete_bbox,
    load_bboxes,
    load_classes,
    resolve_label_for_review,
    save_bboxes,
    update_bbox,
)
from review_status import (
    find_next_uncompleted_index,
    load_completed_image_keys,
    save_completed_image_keys,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_BBOX_RATIO = 0.25
HANDLE_SIZE = 8
MIN_BBOX_IMAGE_SIZE = 4.0
RESIZE_HANDLE_NAMES = ("nw", "ne", "se", "sw", "n", "e", "s", "w")
ZOOM_IN_FACTOR = 1.25
ZOOM_OUT_FACTOR = 0.8
CLASS_BBOX_COLORS = (
    "#00e676",
    "#ff6d00",
    "#2979ff",
    "#ff1744",
    "#d500f9",
    "#00b8d4",
    "#ffd600",
    "#651fff",
    "#1de9b6",
    "#f50057",
    "#76ff03",
    "#ff9100",
)


def class_bbox_color(class_id: int) -> str:
    """class ID에 따라 항상 같은 bbox 표시 색상을 반환한다."""
    return CLASS_BBOX_COLORS[class_id % len(CLASS_BBOX_COLORS)]


class ReviewApp:
    """자동 생성 bbox를 사람이 검수하고 저장하는 로컬 GUI."""

    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.config_path = config_path
        self.base_dir = config_path.parent
        self.config = self.load_config(config_path)

        self.source_dir = self.resolve_path(self.config["source_dir"])
        self.labels_auto_dir = self.resolve_path(self.config["labels_auto_dir"])
        self.labels_reviewed_dir = self.resolve_path(
            self.config.get("labels_reviewed_dir", "labels_reviewed")
        )
        self.review_status_path = self.labels_reviewed_dir / "review_status.json"
        self.completed_image_keys = load_completed_image_keys(self.review_status_path)
        self.dataset_output_dir = self.optional_resolve_path(
            self.config.get("dataset_output_dir")
        )
        self.classes_path = self.resolve_path(self.config.get("classes_path", "classes.txt"))
        self.classes = load_classes(self.classes_path)
        self.image_paths = self.find_images(self.source_dir)
        self.known_image_paths = set(self.image_paths)
        self.watch_source_dir = bool(self.config.get("watch_source_dir", False))
        self.model_specs = build_model_specs(self.config, self.base_dir)
        self.duplicate_iou = normalize_threshold(
            self.config["duplicate_iou"],
            "duplicate_iou",
        )
        self.loaded_models = None

        self.current_image_index = 0
        self.current_image_path: Path | None = None
        self.current_image_width = 0
        self.current_image_height = 0
        self.original_image = None
        self.display_scale = 1.0
        self.fit_scale = 1.0
        self.photo_image: tk.PhotoImage | None = None
        self.bboxes: list[BBox] = []
        self.bbox_history = BBoxHistory()
        self.selected_bbox_index: int | None = None
        self.updating_bbox_list = False
        self.edit_mode: str | None = None
        self.edit_start: tuple[float, float] | None = None
        self.edit_original_rect: tuple[float, float, float, float] | None = None
        self.edit_handle: str | None = None
        self.edit_original_bboxes: list[BBox] | None = None

        self.root.title("YOLO bbox 검수")
        self.build_layout()
        self.load_current_image()
        self.schedule_source_watch()

    def build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Button(toolbar, text="이미지 폴더 선택", command=self.choose_source_dir).pack(side="left")
        ttk.Button(toolbar, text="드론 폴더 감시", command=self.choose_watched_source_dir).pack(side="left")
        ttk.Button(toolbar, text="모델 파일 선택", command=self.choose_model_files).pack(side="left")
        ttk.Button(toolbar, text="학습 폴더 선택", command=self.choose_dataset_output_dir).pack(side="left")
        ttk.Button(toolbar, text="새로고침", command=self.reload_images).pack(side="left")
        ttk.Button(toolbar, text="이전", command=self.previous_image).pack(side="left")
        ttk.Button(toolbar, text="다음", command=self.next_image).pack(side="left")
        ttk.Button(toolbar, text="다음 미검수", command=self.next_unreviewed_image).pack(side="left")
        ttk.Button(toolbar, text="저장", command=self.save_reviewed).pack(side="left")
        self.review_complete_button = ttk.Button(
            toolbar,
            text="검수 완료",
            command=self.toggle_review_complete,
        )
        self.review_complete_button.pack(side="left")
        ttk.Button(toolbar, text="bbox 추가", command=self.add_default_bbox).pack(side="left")
        ttk.Button(toolbar, text="삭제", command=self.delete_selected_bbox).pack(side="left")
        ttk.Button(toolbar, text="class 변경", command=self.change_selected_class).pack(side="left")
        ttk.Button(toolbar, text="class 추가", command=self.add_class).pack(side="left")

        self.status_label = ttk.Label(toolbar, text="")
        self.status_label.pack(side="left", padx=12)

        self.class_var = tk.StringVar(value=self.classes[0] if self.classes else "")
        self.class_box = ttk.Combobox(
            toolbar,
            textvariable=self.class_var,
            values=self.classes,
            state="readonly",
            width=24,
        )
        self.class_box.pack(side="right")
        ttk.Label(toolbar, text="추가/변경 class").pack(side="right", padx=6)

        self.canvas = tk.Canvas(main, background="#1f1f1f", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.delete_bbox_at_mouse)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.root.bind("<Delete>", self.delete_selected_bbox)
        self.root.bind("<BackSpace>", self.delete_selected_bbox)
        self.root.bind("<Control-z>", self.undo_bbox_change)
        self.root.bind("<Control-y>", self.redo_bbox_change)

        side = ttk.Frame(main, padding=(8, 8, 0, 0))
        side.grid(row=1, column=1, sticky="ns")
        side.rowconfigure(0, weight=1)

        self.bbox_list = tk.Listbox(side, width=36, height=24)
        self.bbox_list.grid(row=0, column=0, sticky="ns")
        self.bbox_list.bind("<<ListboxSelect>>", self.on_bbox_select)

    def load_current_image(self) -> None:
        if not self.image_paths:
            self.current_image_path = None
            self.bboxes = []
            self.bbox_history.reset()
            self.selected_bbox_index = None
            self.canvas.delete("all")
            self.bbox_list.delete(0, tk.END)
            self.status_label.config(text=f"이미지가 없습니다: {self.source_dir}")
            return

        self.current_image_path = self.image_paths[self.current_image_index]
        self.bbox_history.reset()
        self.edit_original_bboxes = None
        image = cv2.imread(str(self.current_image_path))
        if image is None:
            messagebox.showerror("이미지 오류", f"이미지를 읽을 수 없습니다: {self.current_image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image_height, self.current_image_width = image.shape[:2]
        self.original_image = image

        max_width = 1000
        max_height = 720
        self.fit_scale = min(
            max_width / self.current_image_width,
            max_height / self.current_image_height,
            1.0,
        )
        self.display_scale = self.fit_scale
        self.render_image(reset_view=True)

        auto_label_path = self.label_path_for(self.current_image_path, self.labels_auto_dir)
        reviewed_label_path = self.label_path_for(
            self.current_image_path,
            self.labels_reviewed_dir,
        )
        try:
            label_path = resolve_label_for_review(
                reviewed_label_path,
                auto_label_path,
                lambda: self.create_auto_label_for_current_image(auto_label_path),
            )
        except Exception as exc:
            messagebox.showerror(
                "자동 라벨 생성 오류",
                (
                    "이 이미지는 아직 자동 txt가 없습니다. 그래서 YOLO 모델로 txt를 새로 만들려고 했지만, "
                    "자동 생성 준비 또는 추론 단계에서 실패했습니다.\n\n"
                    "확인할 것:\n"
                    "1. GUI의 '모델 파일 선택'에서 사용할 모델이 체크되어 있는지\n"
                    "2. 체크된 모델의 .pt 파일 경로가 실제 파일인지\n"
                    "3. 해당 이미지 파일을 OpenCV/YOLO가 읽을 수 있는지\n\n"
                    "실제 오류:\n"
                    f"{exc}"
                ),
            )
            self.bboxes = []
            self.selected_bbox_index = None
            self.render_bboxes()
            self.update_status()
            return

        self.bboxes = load_bboxes(label_path)
        self.selected_bbox_index = None
        self.render_image()
        self.update_status()

    def create_auto_label_for_current_image(self, auto_label_path: Path) -> None:
        """현재 사진의 txt가 없을 때 YOLO 모델로 자동 라벨 파일을 생성한다."""
        if self.current_image_path is None:
            raise RuntimeError("current image is not selected")

        self.status_label.config(text="자동 라벨 생성 중...")
        self.root.update_idletasks()

        if self.loaded_models is None:
            self.loaded_models = load_models(self.model_specs)

        generate_auto_label_file(
            self.current_image_path,
            auto_label_path,
            self.loaded_models,
            self.duplicate_iou,
        )

    def choose_model_files(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("YOLO 모델 파일 선택")
        dialog.transient(self.root)
        dialog.grab_set()

        model_path_vars: list[tk.StringVar] = []
        model_enabled_vars: list[tk.BooleanVar] = []
        model_entries = self.config.get("models", [])

        ttk.Label(dialog, text="사용").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Label(dialog, text="모델").grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(dialog, text=".pt 경로").grid(row=0, column=2, padx=6, pady=6, sticky="w")

        for row_index, model_config in enumerate(model_entries, start=1):
            model_name = str(model_config.get("name", f"model_{row_index}"))
            initial_path = str(model_config.get("path", ""))
            path_var = tk.StringVar(value=initial_path)
            enabled_var = tk.BooleanVar(value=bool(model_config.get("enabled", True)))
            model_path_vars.append(path_var)
            model_enabled_vars.append(enabled_var)

            ttk.Checkbutton(dialog, variable=enabled_var).grid(
                row=row_index,
                column=0,
                padx=6,
                pady=4,
                sticky="w",
            )

            ttk.Label(dialog, text=model_name).grid(
                row=row_index,
                column=1,
                padx=6,
                pady=4,
                sticky="w",
            )
            ttk.Entry(dialog, textvariable=path_var, width=70).grid(
                row=row_index,
                column=2,
                padx=6,
                pady=4,
                sticky="ew",
            )
            ttk.Button(
                dialog,
                text="찾기",
                command=lambda target=path_var: self.browse_model_file(target),
            ).grid(row=row_index, column=3, padx=6, pady=4)

        dialog.columnconfigure(2, weight=1)

        button_row = len(model_path_vars) + 1
        ttk.Button(
            dialog,
            text="적용",
            command=lambda: self.apply_model_files(
                dialog,
                model_path_vars,
                model_enabled_vars,
                save_config=False,
            ),
        ).grid(row=button_row, column=2, padx=6, pady=8, sticky="e")
        ttk.Button(
            dialog,
            text="적용 후 설정 저장",
            command=lambda: self.apply_model_files(
                dialog,
                model_path_vars,
                model_enabled_vars,
                save_config=True,
            ),
        ).grid(row=button_row, column=3, padx=6, pady=8, sticky="e")

    def browse_model_file(self, path_var: tk.StringVar) -> None:
        selected_file = filedialog.askopenfilename(
            title="YOLO .pt 모델 선택",
            initialdir=str(self.base_dir),
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")],
        )
        if selected_file:
            path_var.set(str(Path(selected_file)))

    def apply_model_files(
        self,
        dialog: tk.Toplevel,
        model_path_vars: list[tk.StringVar],
        model_enabled_vars: list[tk.BooleanVar],
        save_config: bool,
    ) -> None:
        try:
            self.update_model_paths(model_path_vars, model_enabled_vars)
            self.model_specs = build_model_specs(self.config, self.base_dir)
        except Exception as exc:
            messagebox.showerror("모델 설정 오류", str(exc))
            return

        self.loaded_models = None

        if save_config:
            try:
                self.save_config()
            except Exception as exc:
                messagebox.showerror("설정 저장 오류", str(exc))
                return

        dialog.destroy()
        self.status_label.config(text="모델 파일 설정이 적용되었습니다.")

    def update_model_paths(
        self,
        model_path_vars: list[tk.StringVar],
        model_enabled_vars: list[tk.BooleanVar],
    ) -> None:
        model_entries = self.config.get("models", [])
        if not isinstance(model_entries, list):
            raise ValueError("models must be a YAML list")
        if len(model_entries) != len(model_path_vars) or len(model_entries) != len(model_enabled_vars):
            raise ValueError("model input count does not match config models")

        enabled_count = 0
        for model_config, path_var, enabled_var in zip(
            model_entries,
            model_path_vars,
            model_enabled_vars,
        ):
            if not isinstance(model_config, dict):
                raise ValueError("each model config must be a mapping")
            model_config["enabled"] = bool(enabled_var.get())
            if not model_config["enabled"]:
                continue

            enabled_count += 1
            selected_path_text = path_var.get().strip()
            if not selected_path_text:
                raise ValueError("enabled model must have a .pt path")
            selected_path = Path(selected_path_text)
            if not selected_path.exists():
                raise FileNotFoundError(f"model_path does not exist: {selected_path}")
            if selected_path.suffix.lower() != ".pt":
                raise ValueError(f"model_path must be a .pt file: {selected_path}")
            model_config["path"] = self.path_for_config(selected_path)

        if enabled_count == 0:
            raise ValueError("at least one model must be enabled")

    def choose_source_dir(self) -> None:
        selected_dir = filedialog.askdirectory(
            title="이미지 폴더 선택",
            initialdir=str(self.source_dir),
        )
        if not selected_dir:
            return

        self.source_dir = Path(selected_dir)
        self.config["source_dir"] = self.path_for_config(self.source_dir)
        self.watch_source_dir = False
        self.config["watch_source_dir"] = False
        self.save_config()
        self.reload_images()

    def choose_watched_source_dir(self) -> None:
        selected_dir = filedialog.askdirectory(
            title="드론 사진이 들어오는 폴더 선택",
            initialdir=str(self.source_dir),
        )
        if not selected_dir:
            return

        self.source_dir = Path(selected_dir)
        self.config["source_dir"] = self.path_for_config(self.source_dir)
        self.watch_source_dir = True
        self.config["watch_source_dir"] = True
        self.save_config()
        self.reload_images()
        self.status_label.config(text=f"드론 폴더 감시 중: {self.source_dir}")

    def choose_dataset_output_dir(self) -> None:
        selected_dir = filedialog.askdirectory(
            title="학습 데이터셋을 저장할 폴더 선택",
            initialdir=str(self.dataset_output_dir or self.base_dir),
        )
        if not selected_dir:
            return

        self.dataset_output_dir = Path(selected_dir)
        self.config["dataset_output_dir"] = self.path_for_config(self.dataset_output_dir)
        self.save_config()
        self.ensure_training_dataset_dirs()
        self.status_label.config(text=f"학습 폴더 지정됨: {self.dataset_output_dir}")

    def reload_images(self) -> None:
        try:
            image_paths = self.find_images(self.source_dir)
        except Exception as exc:
            messagebox.showerror("이미지 폴더 오류", str(exc))
            return

        current_image_path = self.current_image_path
        self.image_paths = image_paths
        self.known_image_paths = set(image_paths)
        if current_image_path in self.image_paths:
            self.current_image_index = self.image_paths.index(current_image_path)
        else:
            self.current_image_index = 0
        self.load_current_image()

    def schedule_source_watch(self) -> None:
        self.root.after(3000, self.poll_source_dir)

    def poll_source_dir(self) -> None:
        try:
            if self.watch_source_dir:
                image_paths = self.find_images(self.source_dir)
                image_path_set = set(image_paths)
                new_image_count = len(image_path_set - self.known_image_paths)
                if new_image_count > 0:
                    current_image_path = self.current_image_path
                    self.image_paths = image_paths
                    self.known_image_paths = image_path_set
                    if current_image_path in self.image_paths:
                        self.current_image_index = self.image_paths.index(current_image_path)
                    elif self.image_paths:
                        self.current_image_index = 0
                        self.load_current_image()
                    self.status_label.config(
                        text=f"새 드론 사진 {new_image_count}장 감지됨: 총 {len(self.image_paths)}장"
                    )
        finally:
            self.schedule_source_watch()

    def render_bboxes(self) -> None:
        self.canvas.delete("bbox")
        self.updating_bbox_list = True
        self.bbox_list.delete(0, tk.END)

        for index, bbox in enumerate(self.bboxes):
            left, top, right, bottom = bbox_to_canvas_rect(
                bbox,
                self.current_image_width,
                self.current_image_height,
            )
            left *= self.display_scale
            top *= self.display_scale
            right *= self.display_scale
            bottom *= self.display_scale

            color = class_bbox_color(bbox.class_id)
            width = 4 if index == self.selected_bbox_index else 2
            self.canvas.create_rectangle(
                left,
                top,
                right,
                bottom,
                outline=color,
                width=width,
                tags=("bbox",),
            )

            class_name = self.classes[bbox.class_id] if 0 <= bbox.class_id < len(self.classes) else str(bbox.class_id)
            self.canvas.create_text(
                left + 4,
                max(top - 10, 10),
                anchor="w",
                text=class_name,
                fill=color,
                tags=("bbox",),
            )
            if index == self.selected_bbox_index:
                self.render_resize_handles(left, top, right, bottom)
            self.bbox_list.insert(tk.END, f"{index}: {class_name}")

        if self.selected_bbox_index is not None and 0 <= self.selected_bbox_index < len(self.bboxes):
            self.bbox_list.selection_set(self.selected_bbox_index)
            self.bbox_list.see(self.selected_bbox_index)
        self.updating_bbox_list = False

    def render_resize_handles(self, left: float, top: float, right: float, bottom: float) -> None:
        """선택된 bbox의 크기를 조절할 수 있는 화면 핸들을 그린다."""
        half_size = HANDLE_SIZE / 2
        for handle_x, handle_y in self.resize_handle_points(left, top, right, bottom).values():
            self.canvas.create_rectangle(
                handle_x - half_size,
                handle_y - half_size,
                handle_x + half_size,
                handle_y + half_size,
                outline="#1f1f1f",
                fill="#ffcc00",
                width=1,
                tags=("bbox", "bbox_handle"),
            )

    def render_image(self, reset_view: bool = False) -> None:
        """현재 배율로 이미지를 다시 그리고 화면 이동 가능 범위를 갱신한다."""
        if self.original_image is None:
            return

        display_width = max(1, int(self.current_image_width * self.display_scale))
        display_height = max(1, int(self.current_image_height * self.display_scale))
        if self.display_scale == 1.0:
            display_image = self.original_image
        else:
            interpolation = cv2.INTER_AREA if self.display_scale < 1.0 else cv2.INTER_LINEAR
            display_image = cv2.resize(
                self.original_image,
                (display_width, display_height),
                interpolation=interpolation,
            )

        self.photo_image = self.cv_image_to_photo_image(display_image)
        self.canvas.config(width=display_width, height=display_height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.canvas.config(scrollregion=(0, 0, display_width, display_height))
        if reset_view:
            self.canvas.xview_moveto(0.0)
            self.canvas.yview_moveto(0.0)
        self.render_bboxes()

    def on_mouse_wheel(self, event: tk.Event) -> None:
        """마우스 위치를 유지하면서 휠 한 단계마다 빠르게 확대·축소한다."""
        if self.current_image_path is None:
            return

        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            zoom_factor = ZOOM_OUT_FACTOR
        else:
            zoom_factor = ZOOM_IN_FACTOR

        minimum_scale = max(self.fit_scale * 0.25, 0.05)
        maximum_scale = 4.0
        next_scale = max(minimum_scale, min(self.display_scale * zoom_factor, maximum_scale))
        if next_scale == self.display_scale:
            return

        image_x = self.canvas.canvasx(event.x) / self.display_scale
        image_y = self.canvas.canvasy(event.y) / self.display_scale
        self.display_scale = next_scale
        self.render_image()
        self.keep_image_point_under_mouse(image_x, image_y, event.x, event.y)
        self.update_status()

    def on_mouse_down(self, event: tk.Event) -> None:
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        handle_name = self.find_resize_handle_at_canvas_point(canvas_x, canvas_y)
        if handle_name is not None:
            self.start_bbox_edit("resize", canvas_x, canvas_y, handle_name)
            return

        clicked_bbox_index = self.find_bbox_at_canvas_point(canvas_x, canvas_y)
        if clicked_bbox_index is not None:
            self.selected_bbox_index = clicked_bbox_index
            self.render_bboxes()
            self.start_bbox_edit("move", canvas_x, canvas_y, None)
            return

        self.selected_bbox_index = None
        self.clear_bbox_edit()
        self.render_bboxes()
        self.edit_mode = "pan"
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.config(cursor="fleur")

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.edit_mode == "pan":
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            return
        if self.selected_bbox_index is None or self.edit_mode is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        if self.edit_mode == "move":
            self.move_selected_bbox(canvas_x, canvas_y)
        elif self.edit_mode == "resize":
            self.resize_selected_bbox(canvas_x, canvas_y)

    def on_mouse_up(self, event: tk.Event) -> None:
        if self.edit_mode is None:
            return

        completed_edit_mode = self.edit_mode
        previous_bboxes = self.edit_original_bboxes
        self.clear_bbox_edit()
        self.canvas.config(cursor="")
        if (
            completed_edit_mode in {"move", "resize"}
            and previous_bboxes is not None
            and previous_bboxes != self.bboxes
        ):
            self.bbox_history.record(previous_bboxes)
            self.mark_current_reviewing()
        self.edit_original_bboxes = None
        self.update_status()

    def keep_image_point_under_mouse(
        self,
        image_x: float,
        image_y: float,
        mouse_x: float,
        mouse_y: float,
    ) -> None:
        """확대 전 마우스 아래 이미지 지점이 확대 후에도 같은 화면 위치에 오게 한다."""
        self.canvas.update_idletasks()
        scroll_width = max(1.0, self.current_image_width * self.display_scale)
        scroll_height = max(1.0, self.current_image_height * self.display_scale)
        viewport_width = max(1.0, float(self.canvas.winfo_width()))
        viewport_height = max(1.0, float(self.canvas.winfo_height()))
        left = image_x * self.display_scale - mouse_x
        top = image_y * self.display_scale - mouse_y
        maximum_left = max(0.0, scroll_width - viewport_width)
        maximum_top = max(0.0, scroll_height - viewport_height)
        left = max(0.0, min(left, maximum_left))
        top = max(0.0, min(top, maximum_top))
        self.canvas.xview_moveto(left / scroll_width)
        self.canvas.yview_moveto(top / scroll_height)

    def add_default_bbox(self) -> None:
        """선택된 class로 이미지 중앙에 기본 크기 bbox를 추가한다."""
        if self.current_image_path is None:
            return

        bbox_width = min(
            self.current_image_width,
            max(MIN_BBOX_IMAGE_SIZE, self.current_image_width * DEFAULT_BBOX_RATIO),
        )
        bbox_height = min(
            self.current_image_height,
            max(MIN_BBOX_IMAGE_SIZE, self.current_image_height * DEFAULT_BBOX_RATIO),
        )
        left = (self.current_image_width - bbox_width) / 2
        top = (self.current_image_height - bbox_height) / 2
        bbox = bbox_from_canvas_rect(
            self.current_class_id(),
            left,
            top,
            left + bbox_width,
            top + bbox_height,
            self.current_image_width,
            self.current_image_height,
        )
        self.bbox_history.record(self.bboxes)
        self.bboxes = create_bbox(self.bboxes, bbox)
        self.mark_current_reviewing()
        self.selected_bbox_index = len(self.bboxes) - 1
        self.render_bboxes()
        self.update_status()

    def start_bbox_edit(
        self,
        edit_mode: str,
        canvas_x: float,
        canvas_y: float,
        handle_name: str | None,
    ) -> None:
        if self.selected_bbox_index is None:
            return

        self.edit_original_bboxes = list(self.bboxes)
        self.edit_mode = edit_mode
        self.edit_start = (canvas_x, canvas_y)
        self.edit_original_rect = bbox_to_canvas_rect(
            self.bboxes[self.selected_bbox_index],
            self.current_image_width,
            self.current_image_height,
        )
        self.edit_handle = handle_name

    def clear_bbox_edit(self) -> None:
        self.edit_mode = None
        self.edit_start = None
        self.edit_original_rect = None
        self.edit_handle = None

    def move_selected_bbox(self, canvas_x: float, canvas_y: float) -> None:
        if self.selected_bbox_index is None or self.edit_start is None or self.edit_original_rect is None:
            return

        start_x, start_y = self.edit_start
        left, top, right, bottom = self.edit_original_rect
        delta_x = (canvas_x - start_x) / self.display_scale
        delta_y = (canvas_y - start_y) / self.display_scale
        next_left = left + delta_x
        next_top = top + delta_y
        next_right = right + delta_x
        next_bottom = bottom + delta_y

        if next_left < 0:
            next_right -= next_left
            next_left = 0
        if next_top < 0:
            next_bottom -= next_top
            next_top = 0
        if next_right > self.current_image_width:
            overflow = next_right - self.current_image_width
            next_left -= overflow
            next_right = self.current_image_width
        if next_bottom > self.current_image_height:
            overflow = next_bottom - self.current_image_height
            next_top -= overflow
            next_bottom = self.current_image_height

        self.update_selected_bbox_from_image_rect(next_left, next_top, next_right, next_bottom)

    def resize_selected_bbox(self, canvas_x: float, canvas_y: float) -> None:
        if (
            self.selected_bbox_index is None
            or self.edit_original_rect is None
            or self.edit_handle is None
        ):
            return

        left, top, right, bottom = self.edit_original_rect
        pointer_x = max(0.0, min(canvas_x / self.display_scale, float(self.current_image_width)))
        pointer_y = max(0.0, min(canvas_y / self.display_scale, float(self.current_image_height)))
        minimum_size = min(
            MIN_BBOX_IMAGE_SIZE,
            float(self.current_image_width),
            float(self.current_image_height),
        )

        if "w" in self.edit_handle:
            left = max(0.0, min(pointer_x, right - minimum_size))
        if "e" in self.edit_handle:
            right = min(float(self.current_image_width), max(pointer_x, left + minimum_size))
        if "n" in self.edit_handle:
            top = max(0.0, min(pointer_y, bottom - minimum_size))
        if "s" in self.edit_handle:
            bottom = min(float(self.current_image_height), max(pointer_y, top + minimum_size))

        self.update_selected_bbox_from_image_rect(left, top, right, bottom)

    def update_selected_bbox_from_image_rect(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> None:
        if self.selected_bbox_index is None:
            return

        current_bbox = self.bboxes[self.selected_bbox_index]
        next_bbox = bbox_from_canvas_rect(
            current_bbox.class_id,
            left,
            top,
            right,
            bottom,
            self.current_image_width,
            self.current_image_height,
        )
        self.bboxes = update_bbox(self.bboxes, self.selected_bbox_index, next_bbox)
        self.render_bboxes()

    def on_bbox_select(self, _event: tk.Event) -> None:
        if self.updating_bbox_list:
            return

        selection = self.bbox_list.curselection()
        if not selection:
            self.selected_bbox_index = None
        else:
            self.selected_bbox_index = int(selection[0])
        self.render_bboxes()

    def delete_selected_bbox(self, _event: tk.Event | None = None) -> None:
        if self.selected_bbox_index is None:
            return
        self.bbox_history.record(self.bboxes)
        self.bboxes = delete_bbox(self.bboxes, self.selected_bbox_index)
        self.mark_current_reviewing()
        self.selected_bbox_index = None
        self.render_bboxes()
        self.update_status()

    def delete_bbox_at_mouse(self, event: tk.Event) -> None:
        bbox_index = self.find_bbox_at_canvas_point(
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y),
        )
        if bbox_index is None:
            return
        self.selected_bbox_index = bbox_index
        self.delete_selected_bbox()

    def find_resize_handle_at_canvas_point(self, canvas_x: float, canvas_y: float) -> str | None:
        if self.selected_bbox_index is None:
            return None

        left, top, right, bottom = bbox_to_canvas_rect(
            self.bboxes[self.selected_bbox_index],
            self.current_image_width,
            self.current_image_height,
        )
        left *= self.display_scale
        top *= self.display_scale
        right *= self.display_scale
        bottom *= self.display_scale

        tolerance = HANDLE_SIZE
        for handle_name in RESIZE_HANDLE_NAMES:
            handle_x, handle_y = self.resize_handle_points(left, top, right, bottom)[handle_name]
            if abs(canvas_x - handle_x) <= tolerance and abs(canvas_y - handle_y) <= tolerance:
                return handle_name
        return None

    def find_bbox_at_canvas_point(self, canvas_x: float, canvas_y: float) -> int | None:
        for index in range(len(self.bboxes) - 1, -1, -1):
            left, top, right, bottom = bbox_to_canvas_rect(
                self.bboxes[index],
                self.current_image_width,
                self.current_image_height,
            )
            left *= self.display_scale
            top *= self.display_scale
            right *= self.display_scale
            bottom *= self.display_scale
            if left <= canvas_x <= right and top <= canvas_y <= bottom:
                return index
        return None

    @staticmethod
    def resize_handle_points(
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> dict[str, tuple[float, float]]:
        middle_x = (left + right) / 2
        middle_y = (top + bottom) / 2
        return {
            "nw": (left, top),
            "ne": (right, top),
            "se": (right, bottom),
            "sw": (left, bottom),
            "n": (middle_x, top),
            "e": (right, middle_y),
            "s": (middle_x, bottom),
            "w": (left, middle_y),
        }

    def change_selected_class(self) -> None:
        if self.selected_bbox_index is None:
            return
        self.bbox_history.record(self.bboxes)
        self.bboxes = change_bbox_class(
            self.bboxes,
            self.selected_bbox_index,
            self.current_class_id(),
        )
        self.mark_current_reviewing()
        self.render_bboxes()
        self.update_status()

    def undo_bbox_change(self, _event: tk.Event | None = None) -> None:
        """Ctrl+Z 입력 시 현재 이미지의 직전 bbox 목록을 복원한다."""
        previous_bboxes = self.bbox_history.undo(self.bboxes)
        if previous_bboxes is None:
            return
        self.bboxes = previous_bboxes
        self.selected_bbox_index = None
        self.clear_bbox_edit()
        self.edit_original_bboxes = None
        self.mark_current_reviewing()
        self.render_bboxes()
        self.update_status()

    def redo_bbox_change(self, _event: tk.Event | None = None) -> None:
        """Ctrl+Y 입력 시 취소했던 bbox 변경을 다시 적용한다."""
        restored_bboxes = self.bbox_history.redo(self.bboxes)
        if restored_bboxes is None:
            return
        self.bboxes = restored_bboxes
        self.selected_bbox_index = None
        self.clear_bbox_edit()
        self.edit_original_bboxes = None
        self.mark_current_reviewing()
        self.render_bboxes()
        self.update_status()

    def add_class(self) -> None:
        class_name = simpledialog.askstring(
            "class 추가",
            "추가할 class 이름을 입력하세요.",
            parent=self.root,
        )
        if class_name is None:
            return

        class_name = class_name.strip()
        if not class_name:
            messagebox.showerror("class 추가 오류", "class 이름은 비어 있을 수 없습니다.")
            return
        if any(existing_class == class_name for existing_class in self.classes):
            messagebox.showerror("class 추가 오류", f"이미 존재하는 class입니다: {class_name}")
            return

        self.classes.append(class_name)
        self.config["classes"] = list(self.classes)
        self.class_box.config(values=self.classes)
        self.class_var.set(class_name)
        self.save_classes()
        self.save_training_classes()
        self.save_config()
        self.status_label.config(text=f"class 추가됨: {class_name}")

    def save_reviewed(self) -> None:
        if self.save_current_reviewed_state():
            self.update_status(saved=True)

    def toggle_review_complete(self) -> None:
        """현재 이미지를 검수 완료로 표시하거나 다시 검수 중으로 되돌린다."""
        if self.current_image_path is None:
            return

        image_key = self.current_image_key()
        if image_key in self.completed_image_keys:
            self.completed_image_keys.remove(image_key)
            if self.save_review_status():
                self.update_status()
            return

        if not self.save_current_reviewed_state():
            return
        self.completed_image_keys.add(image_key)
        if self.save_review_status():
            self.update_status(saved=True)

    def mark_current_reviewing(self) -> None:
        """완료된 이미지를 다시 편집하면 검수 완료 표시를 해제한다."""
        if self.current_image_path is None:
            return
        image_key = self.current_image_key()
        if image_key not in self.completed_image_keys:
            return
        self.completed_image_keys.remove(image_key)
        self.save_review_status()

    def save_review_status(self) -> bool:
        """메모리의 검수 완료 목록을 review_status.json에 저장한다."""
        try:
            save_completed_image_keys(
                self.review_status_path,
                self.completed_image_keys,
            )
        except Exception as exc:
            messagebox.showerror(
                "검수 상태 저장 오류",
                f"검수 완료 상태를 저장하지 못했습니다.\n\n실제 오류:\n{exc}",
            )
            return False
        return True

    def save_current_reviewed_state(self) -> bool:
        """현재 이미지의 bbox 목록을 검수 txt와 학습 폴더에 저장한다."""
        if self.current_image_path is None:
            return True

        self.clear_bbox_edit()
        try:
            reviewed_label_path = self.label_path_for(
                self.current_image_path,
                self.labels_reviewed_dir,
            )
            save_bboxes(reviewed_label_path, self.bboxes)
            if self.dataset_output_dir is not None:
                self.export_current_training_pair()
        except Exception as exc:
            messagebox.showerror(
                "저장 오류",
                (
                    "현재 이미지의 bbox를 저장하지 못했습니다.\n"
                    "저장에 실패했기 때문에 다른 이미지로 이동하지 않습니다.\n\n"
                    f"실제 오류:\n{exc}"
                ),
            )
            return False

        return True

    def export_current_training_pair(self) -> None:
        if self.current_image_path is None:
            return
        if self.dataset_output_dir is None:
            return

        image_output_path, label_output_path = self.training_pair_paths(
            self.current_image_path
        )
        image_output_path.parent.mkdir(parents=True, exist_ok=True)
        label_output_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self.current_image_path, image_output_path)
        save_bboxes(label_output_path, self.bboxes)
        self.save_training_classes()

    def training_pair_paths(self, image_path: Path) -> tuple[Path, Path]:
        if self.dataset_output_dir is None:
            raise RuntimeError("dataset output dir is not selected")
        relative_image_path = image_path.relative_to(self.source_dir)
        image_output_path = self.dataset_output_dir / "images" / "train" / relative_image_path
        label_output_path = (
            self.dataset_output_dir
            / "labels"
            / "train"
            / relative_image_path.with_suffix(".txt")
        )
        return image_output_path, label_output_path

    def ensure_training_dataset_dirs(self) -> None:
        if self.dataset_output_dir is None:
            return
        (self.dataset_output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.dataset_output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        self.save_training_classes()

    def save_training_classes(self) -> None:
        if self.dataset_output_dir is None:
            return
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_output_dir / "classes.txt").write_text(
            "\n".join(self.classes) + "\n",
            encoding="utf-8",
        )

    def previous_image(self) -> None:
        if not self.image_paths:
            return
        self.navigate_to_image(max(0, self.current_image_index - 1))

    def next_image(self) -> None:
        if not self.image_paths:
            return
        self.navigate_to_image(min(len(self.image_paths) - 1, self.current_image_index + 1))

    def next_unreviewed_image(self) -> None:
        """현재 bbox를 저장한 뒤 다음 미검수 이미지를 순환 검색하여 연다."""
        if not self.image_paths:
            return
        if not self.save_current_reviewed_state():
            return

        image_keys = [self.image_key_for(image_path) for image_path in self.image_paths]
        next_image_index = find_next_uncompleted_index(
            image_keys,
            self.completed_image_keys,
            self.current_image_index,
        )
        if next_image_index is None:
            messagebox.showinfo("검수 완료", "모든 이미지의 검수가 완료되었습니다.")
            self.update_status(saved=True)
            return
        if next_image_index == self.current_image_index:
            self.update_status(saved=True)
            return

        self.current_image_index = next_image_index
        self.load_current_image()

    def navigate_to_image(self, next_image_index: int) -> None:
        """이미지를 이동하기 전에 현재 검수 결과를 먼저 저장한다."""
        if not self.save_current_reviewed_state():
            return

        if next_image_index == self.current_image_index:
            self.update_status(saved=True)
            return

        self.current_image_index = next_image_index
        self.load_current_image()

    def current_class_id(self) -> int:
        class_name = self.class_var.get()
        if class_name not in self.classes:
            return 0
        return self.classes.index(class_name)

    def update_status(self, saved: bool = False) -> None:
        if self.current_image_path is None:
            return

        image_key = self.current_image_key()
        reviewed_label_path = self.label_path_for(
            self.current_image_path,
            self.labels_reviewed_dir,
        )
        if image_key in self.completed_image_keys:
            status = "검수 완료"
            self.review_complete_button.config(text="검수 재개")
        elif reviewed_label_path.exists():
            status = "검수 중"
            self.review_complete_button.config(text="검수 완료")
        else:
            status = "자동 생성"
            self.review_complete_button.config(text="검수 완료")

        completed_count = sum(
            1
            for image_path in self.image_paths
            if self.image_key_for(image_path) in self.completed_image_keys
        )
        saved_text = " | 저장됨" if saved else ""
        self.status_label.config(
            text=(
                f"{status}{saved_text} | 완료 {completed_count}/{len(self.image_paths)} "
                f"| 현재 {self.current_image_index + 1}/{len(self.image_paths)} "
                f"| {self.current_image_path.name} | bbox {len(self.bboxes)}개 "
                f"| 확대 {self.display_scale:.2f}x"
            )
        )

    def current_image_key(self) -> str:
        if self.current_image_path is None:
            raise RuntimeError("current image is not selected")
        return self.image_key_for(self.current_image_path)

    def image_key_for(self, image_path: Path) -> str:
        """이미지 경로를 검수 상태 저장용 상대 경로로 변환한다."""
        return image_path.relative_to(self.source_dir).as_posix()

    def label_path_for(self, image_path: Path, label_dir: Path) -> Path:
        relative_image_path = image_path.relative_to(self.source_dir)
        return label_dir / relative_image_path.with_suffix(".txt")

    def resolve_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def optional_resolve_path(self, path_value: Any) -> Path | None:
        if path_value is None or str(path_value).strip() == "":
            return None
        return self.resolve_path(path_value)

    def path_for_config(self, path: Path) -> str:
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(self.base_dir).as_posix()
        except ValueError:
            return str(resolved_path)

    def save_config(self) -> None:
        self.config_path.write_text(
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def save_classes(self) -> None:
        self.classes_path.parent.mkdir(parents=True, exist_ok=True)
        self.classes_path.write_text("\n".join(self.classes) + "\n", encoding="utf-8")

    @staticmethod
    def load_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"config_path does not exist: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config must be a YAML mapping")
        return config

    @staticmethod
    def find_images(source_dir: Path) -> list[Path]:
        if not source_dir.exists():
            raise FileNotFoundError(f"source_dir does not exist: {source_dir}")
        return sorted(
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def cv_image_to_photo_image(image) -> tk.PhotoImage:
        success, encoded_image = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("failed to encode image for Tkinter display")
        encoded_data = base64.b64encode(encoded_image.tobytes()).decode("ascii")
        return tk.PhotoImage(data=encoded_data, format="PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO txt bbox를 로컬 GUI에서 검수합니다.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("autolabel_config.yaml"),
        help="autolabel_config.yaml 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    ReviewApp(root, args.config.resolve())
    root.mainloop()


if __name__ == "__main__":
    main()
