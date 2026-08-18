from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
from typing import Any
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk

import cv2
import yaml

from auto_label import (
    build_model_specs,
    find_missing_model_paths,
    generate_auto_label_file,
    load_models,
    normalize_threshold,
)
from image_io import IMAGE_EXTENSIONS, read_image
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
    DEFAULT_SOURCE_KEY,
    find_next_uncompleted_index,
    load_status_by_source,
    save_status_by_source,
    source_key_for,
)


DEFAULT_BBOX_RATIO = 0.25
HANDLE_SIZE = 8
MIN_BBOX_IMAGE_SIZE = 4.0
MIN_NEW_BBOX_DRAG_PIXELS = 5.0
RESIZE_HANDLE_NAMES = ("nw", "ne", "se", "sw", "n", "e", "s", "w")
ZOOM_IN_FACTOR = 1.25
ZOOM_OUT_FACTOR = 0.8
MAXIMUM_ZOOM_SCALE = 8.0
CANVAS_RESIZE_DEBOUNCE_MS = 120
SOURCE_WATCH_INTERVAL_MS = 3000

# 어두운 단일 테마 — 사진 검수 도구라서 화면의 주인공은 항상 사진이다.
UI = {
    "bg": "#14161b",          # 창 바탕
    "panel": "#1b1e25",       # 도구 모음·상태바 패널
    "control": "#262b35",     # 버튼·입력 필드
    "hover": "#313847",       # 버튼 hover
    "pressed": "#1f242c",     # 버튼 눌림
    "field": "#101216",       # 목록·입력 배경
    "canvas": "#0d0f13",      # 사진 뒤 여백
    "line": "#2c313c",        # 경계선
    "text": "#dce1ea",        # 본문 글자
    "muted": "#8a93a3",       # 보조 글자
    "accent": "#4f8cff",      # 주 동작·선택·진행률
    "accent_press": "#3f74d6",
    "state_auto": "#5a6372",      # 자동 생성
    "state_reviewing": "#e0a63c", # 검수 중
    "state_done": "#3fc57a",      # 검수 완료
    "state_unsaved": "#ff6473",   # 저장 안 됨
}

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
SHORTCUT_HELP_ROWS = (
    ("← / A", "이전 사진"),
    ("→ / D", "다음 사진"),
    ("Enter", "검수 완료 + 다음"),
    ("Ctrl+S", "저장"),
    ("N", "가운데에 bbox 추가"),
    ("Del", "선택 bbox 삭제"),
    ("Ctrl+Z / Y", "실행 취소 / 다시 실행"),
    ("1~9, 0", "class 빠른 선택"),
    ("휠", "확대 / 축소"),
    ("F", "화면에 맞추기"),
    ("가운데 드래그", "화면 이동"),
    ("빈 곳 드래그", "새 bbox 그리기"),
    ("오른쪽 클릭", "bbox 삭제"),
)
RESIZE_CURSORS = {
    "nw": "size_nw_se",
    "se": "size_nw_se",
    "ne": "size_ne_sw",
    "sw": "size_ne_sw",
    "n": "size_ns",
    "s": "size_ns",
    "e": "size_we",
    "w": "size_we",
}


def class_bbox_color(class_id: int) -> str:
    """class ID에 따라 항상 같은 bbox 표시 색상을 반환한다."""
    return CLASS_BBOX_COLORS[class_id % len(CLASS_BBOX_COLORS)]


def contrast_text_color(background_hex: str) -> str:
    """칩 배경색 위에서 읽히는 글자색(어두운색/밝은색)을 고른다."""
    red = int(background_hex[1:3], 16)
    green = int(background_hex[3:5], 16)
    blue = int(background_hex[5:7], 16)
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    return "#10131a" if luminance > 0.5 else "#f2f4f8"


def enable_windows_dpi_awareness() -> None:
    """Windows 배율(125% 등)에서 창이 비트맵 확대되어 흐려지는 것을 막는다.

    tk.Tk()를 만들기 전에 불러야 한다. 실패해도 앱 동작에는 지장이 없다.
    """
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


class ReviewApp:
    """자동 생성 bbox를 사람이 검수하고 저장하는 로컬 GUI."""

    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.config_path = config_path
        self.base_dir = config_path.parent
        self.config = self.load_config(config_path)
        self.startup_warnings: list[str] = []
        self.apply_display_scaling()

        self.source_dir = self.resolve_path(self.config["source_dir"])
        self.labels_auto_dir = self.resolve_path(self.config["labels_auto_dir"])
        self.labels_reviewed_dir = self.resolve_path(
            self.config.get("labels_reviewed_dir", "labels_reviewed")
        )
        self.review_status_path = self.labels_reviewed_dir / "review_status.json"
        self.status_by_source, status_warning = load_status_by_source(
            self.review_status_path
        )
        if status_warning:
            self.startup_warnings.append(status_warning)
        self.status_source_key = source_key_for(self.source_dir)
        # 폴더 구분이 없던 옛(v1) 상태 파일의 완료 표시를 현재 폴더로 승계한다.
        if (
            self.status_source_key not in self.status_by_source
            and DEFAULT_SOURCE_KEY in self.status_by_source
        ):
            self.status_by_source[self.status_source_key] = self.status_by_source.pop(
                DEFAULT_SOURCE_KEY
            )
        self.completed_image_keys = set(
            self.status_by_source.get(self.status_source_key, set())
        )
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
        self.image_load_failed = False
        self.display_scale = 1.0
        self.fit_scale = 1.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0
        self.image_origin_x = 0.0
        self.image_origin_y = 0.0
        self.photo_image: tk.PhotoImage | None = None
        self.bboxes: list[BBox] = []
        self.bbox_history = BBoxHistory()
        self.selected_bbox_index: int | None = None
        self.updating_bbox_list = False
        self.has_unsaved_changes = False
        self.edit_mode: str | None = None
        self.edit_start: tuple[float, float] | None = None
        self.edit_original_rect: tuple[float, float, float, float] | None = None
        self.edit_handle: str | None = None
        self.edit_original_bboxes: list[BBox] | None = None
        self.new_bbox_start: tuple[float, float] | None = None
        self.pan_start: tuple[float, float, float, float] | None = None
        self.canvas_resize_job: str | None = None
        self.hover_cursor = ""

        self.root.title("YOLO bbox 검수")
        self.apply_theme()
        self.apply_initial_geometry()
        self.build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.update_idletasks()
        self.collect_model_warnings()
        self.load_current_image()
        self.schedule_source_watch()
        self.show_startup_warnings()

    # ------------------------------------------------------------------ 테마

    def apply_display_scaling(self) -> None:
        """모니터 DPI에 맞춰 글자·좌표 배율을 정한다."""
        try:
            dots_per_inch = float(self.root.winfo_fpixels("1i"))
        except tk.TclError:
            dots_per_inch = 96.0
        self.ui_scale = max(1.0, dots_per_inch / 96.0)
        try:
            self.root.tk.call("tk", "scaling", dots_per_inch / 72.0)
        except tk.TclError:
            pass

    def px(self, value: float) -> int:
        """논리 픽셀 값을 현재 모니터 배율의 실제 픽셀로 바꾼다."""
        return max(1, round(value * self.ui_scale))

    def apply_theme(self) -> None:
        """구식 기본 위젯 대신 평면 다크 테마를 입힌다."""
        self.root.configure(bg=UI["bg"])

        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(font_name).configure(family="Malgun Gothic", size=10)
            except tk.TclError:
                pass
        self.canvas_chip_font = tkfont.Font(family="Malgun Gothic", size=9)
        self.chip_font = tkfont.Font(family="Malgun Gothic", size=9, weight="bold")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=UI["panel"], foreground=UI["text"],
                        bordercolor=UI["line"], focuscolor=UI["accent"])
        style.configure("TFrame", background=UI["panel"])
        style.configure("Bg.TFrame", background=UI["bg"])
        style.configure("TLabel", background=UI["panel"], foreground=UI["text"])
        style.configure("Muted.TLabel", background=UI["panel"], foreground=UI["muted"])
        style.configure("Value.TLabel", background=UI["panel"], foreground=UI["text"])

        button_padding = (self.px(10), self.px(5))
        style.configure(
            "TButton",
            background=UI["control"],
            foreground=UI["text"],
            bordercolor=UI["line"],
            lightcolor=UI["control"],
            darkcolor=UI["control"],
            relief="flat",
            padding=button_padding,
        )
        style.map(
            "TButton",
            background=[("pressed", UI["pressed"]), ("active", UI["hover"])],
            foreground=[("disabled", UI["muted"])],
        )
        style.configure(
            "Accent.TButton",
            background=UI["accent"],
            foreground="#0d1117",
            bordercolor=UI["accent"],
            lightcolor=UI["accent"],
            darkcolor=UI["accent"],
            padding=button_padding,
        )
        style.map(
            "Accent.TButton",
            background=[("pressed", UI["accent_press"]), ("active", "#6b9dff")],
            foreground=[("pressed", "#0d1117"), ("active", "#0d1117")],
        )

        style.configure(
            "TCheckbutton",
            background=UI["panel"],
            foreground=UI["text"],
        )
        style.map("TCheckbutton", background=[("active", UI["panel"])])

        style.configure(
            "TEntry",
            fieldbackground=UI["field"],
            foreground=UI["text"],
            bordercolor=UI["line"],
            insertcolor=UI["text"],
        )
        style.configure(
            "TCombobox",
            fieldbackground=UI["field"],
            background=UI["control"],
            foreground=UI["text"],
            bordercolor=UI["line"],
            arrowcolor=UI["muted"],
            selectbackground=UI["field"],
            selectforeground=UI["text"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", UI["field"])],
            background=[("readonly", UI["control"]), ("active", UI["hover"])],
            foreground=[("readonly", UI["text"])],
            # readonly 콤보는 텍스트를 '선택된 상태'로 그리므로 이 두 색까지
            # 지정해야 입력 필드가 흰색으로 남지 않는다.
            selectbackground=[("readonly", UI["field"])],
            selectforeground=[("readonly", UI["text"])],
        )
        self.root.option_add("*TCombobox*Listbox.background", UI["field"])
        self.root.option_add("*TCombobox*Listbox.foreground", UI["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", UI["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#0d1117")

        for orientation in ("Horizontal", "Vertical"):
            style.configure(
                f"{orientation}.TScrollbar",
                background=UI["control"],
                troughcolor=UI["bg"],
                bordercolor=UI["bg"],
                lightcolor=UI["control"],
                darkcolor=UI["control"],
                arrowcolor=UI["muted"],
                relief="flat",
            )
            style.map(
                f"{orientation}.TScrollbar",
                background=[("active", UI["hover"]), ("pressed", UI["hover"])],
            )

        style.configure(
            "Slim.Horizontal.TProgressbar",
            troughcolor=UI["control"],
            background=UI["accent"],
            bordercolor=UI["panel"],
            lightcolor=UI["accent"],
            darkcolor=UI["accent"],
            thickness=self.px(6),
        )

        style.configure(
            "TLabelframe",
            background=UI["panel"],
            bordercolor=UI["line"],
            lightcolor=UI["panel"],
            darkcolor=UI["panel"],
        )
        style.configure(
            "TLabelframe.Label",
            background=UI["panel"],
            foreground=UI["muted"],
        )
        style.configure("TSeparator", background=UI["line"])

    def apply_initial_geometry(self) -> None:
        """모니터 크기에 맞춰 처음 창 크기를 정한다."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = max(self.px(900), int(screen_width * 0.85))
        window_height = max(self.px(640), int(screen_height * 0.85))
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(self.px(900), self.px(600))

    def collect_model_warnings(self) -> None:
        """사용 설정된 모델 중 .pt 파일이 없는 것을 시작 시점에 미리 알린다."""
        missing_models = find_missing_model_paths(self.model_specs)
        if not missing_models:
            return
        missing_text = "\n".join(
            f"  - {model_spec.name}: {model_spec.path}" for model_spec in missing_models
        )
        self.startup_warnings.append(
            "아래 모델 파일을 찾을 수 없습니다. 자동 라벨 생성이 실패합니다.\n"
            f"{missing_text}\n\n"
            "'모델 설정'에서 경로를 고치거나 해당 모델의 체크를 해제하세요."
        )

    def show_startup_warnings(self) -> None:
        """시작할 때 모아둔 경고를 한 번에 보여준다."""
        if not self.startup_warnings:
            return
        messagebox.showwarning("확인이 필요합니다", "\n\n".join(self.startup_warnings))
        self.startup_warnings.clear()

    # ------------------------------------------------------------------ 레이아웃

    def build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=self.px(8))
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        self.build_toolbar(main)
        self.build_canvas(main)
        self.build_side_panel(main)
        self.build_status_bar(main)
        self.bind_shortcuts()

    def toolbar_button(
        self,
        parent: ttk.Frame,
        text: str,
        command,
        accent: bool = False,
    ) -> ttk.Button:
        """포커스를 뺏지 않는 도구 버튼을 만든다(단축키가 계속 살아 있게)."""
        button = ttk.Button(
            parent,
            text=text,
            command=command,
            style="Accent.TButton" if accent else "TButton",
            takefocus=0,
        )
        button.pack(side="left", padx=(0, self.px(4)))
        return button

    def toolbar_separator(self, parent: ttk.Frame) -> None:
        ttk.Separator(parent, orient="vertical").pack(
            side="left", fill="y", padx=self.px(6), pady=self.px(2)
        )

    def build_toolbar(self, main: ttk.Frame) -> None:
        """설정 행과 검수 행으로 나눈 도구 모음을 만든다."""
        setup_row = ttk.Frame(main)
        setup_row.grid(row=0, column=0, columnspan=3, sticky="ew")

        self.toolbar_button(setup_row, "이미지 폴더", self.choose_source_dir)
        self.toolbar_button(setup_row, "폴더 감시", self.choose_watched_source_dir)
        self.toolbar_button(setup_row, "모델 설정", self.choose_model_files)
        self.toolbar_button(setup_row, "학습 폴더", self.choose_dataset_output_dir)
        self.toolbar_separator(setup_row)
        self.toolbar_button(setup_row, "새로고침", self.reload_images)

        # class 선택은 오른쪽에: 색 견본 + 콤보 + 추가 버튼
        class_group = ttk.Frame(setup_row)
        class_group.pack(side="right")
        ttk.Label(class_group, text="class", style="Muted.TLabel").pack(
            side="left", padx=(0, self.px(6))
        )
        swatch_size = self.px(14)
        self.class_swatch = tk.Canvas(
            class_group,
            width=swatch_size,
            height=swatch_size,
            bg=UI["panel"],
            highlightthickness=0,
        )
        self.class_swatch.pack(side="left", padx=(0, self.px(5)))
        self.class_var = tk.StringVar(value=self.classes[0] if self.classes else "")
        self.class_box = ttk.Combobox(
            class_group,
            textvariable=self.class_var,
            values=self.classes,
            state="readonly",
            width=20,
        )
        self.class_box.pack(side="left", padx=(0, self.px(4)))
        self.class_box.bind("<<ComboboxSelected>>", self.on_class_selected)
        self.toolbar_button(class_group, "＋ class", self.add_class)

        review_row = ttk.Frame(main)
        review_row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(self.px(6), 0))

        self.toolbar_button(review_row, "◀ 이전", self.previous_image)
        self.toolbar_button(review_row, "다음 ▶", self.next_image)
        self.toolbar_button(review_row, "다음 미검수", self.next_unreviewed_image)
        self.toolbar_separator(review_row)
        self.toolbar_button(review_row, "저장", self.save_reviewed)
        self.review_complete_button = self.toolbar_button(
            review_row, "검수 완료", self.toggle_review_complete, accent=True
        )
        self.toolbar_separator(review_row)
        self.toolbar_button(review_row, "bbox 추가", self.add_default_bbox)
        self.toolbar_button(review_row, "삭제", self.delete_selected_bbox)
        self.toolbar_button(review_row, "class 변경", self.change_selected_class)
        self.toolbar_separator(review_row)
        self.toolbar_button(review_row, "이 사진 재추론", self.reinfer_current_image)
        self.toolbar_button(review_row, "화면 맞춤", self.fit_image_to_view)

        self.update_class_swatch()

    def build_canvas(self, main: ttk.Frame) -> None:
        """고정 크기 뷰포트 캔버스와 스크롤바를 만든다."""
        canvas_frame = ttk.Frame(main)
        canvas_frame.grid(row=2, column=0, sticky="nsew", pady=(self.px(8), 0))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            background=UI["canvas"],
            highlightthickness=1,
            highlightbackground=UI["line"],
            highlightcolor=UI["line"],
            width=self.px(640),
            height=self.px(480),
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.horizontal_scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient="horizontal",
            command=self.on_horizontal_scroll,
        )
        self.horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.vertical_scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient="vertical",
            command=self.on_vertical_scroll,
        )
        self.vertical_scrollbar.grid(row=0, column=1, sticky="ns")

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<Button-3>", self.delete_bbox_at_mouse)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def build_side_panel(self, main: ttk.Frame) -> None:
        """bbox 목록과 단축키 안내를 담은 오른쪽 패널을 만든다."""
        side = ttk.Frame(main, padding=(self.px(8), self.px(8), 0, 0))
        side.grid(row=2, column=1, sticky="ns")
        side.rowconfigure(1, weight=1)

        self.bbox_list_title = ttk.Label(side, text="bbox 목록", style="Muted.TLabel")
        self.bbox_list_title.grid(row=0, column=0, sticky="w", pady=(0, self.px(4)))
        self.bbox_list = tk.Listbox(
            side,
            width=30,
            height=16,
            exportselection=False,
            bg=UI["field"],
            fg=UI["text"],
            selectbackground="#2f3b52",
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground=UI["line"],
            highlightcolor=UI["accent"],
            relief="flat",
            activestyle="none",
        )
        self.bbox_list.grid(row=1, column=0, sticky="ns")
        self.bbox_list.bind("<<ListboxSelect>>", self.on_bbox_select)

        shortcut_frame = ttk.LabelFrame(
            side, text=" 단축키 ", padding=(self.px(8), self.px(6))
        )
        shortcut_frame.grid(row=2, column=0, sticky="ew", pady=(self.px(8), 0))
        shortcut_frame.columnconfigure(1, weight=1)
        for row_index, (key_text, action_text) in enumerate(SHORTCUT_HELP_ROWS):
            key_chip = tk.Label(
                shortcut_frame,
                text=key_text,
                bg=UI["control"],
                fg=UI["text"],
                font=("Malgun Gothic", 8),
                padx=self.px(5),
                pady=0,
            )
            key_chip.grid(row=row_index, column=0, sticky="w", pady=self.px(1))
            ttk.Label(
                shortcut_frame, text=action_text, style="Muted.TLabel"
            ).grid(row=row_index, column=1, sticky="w", padx=(self.px(8), 0))

    def build_status_bar(self, main: ttk.Frame) -> None:
        """진행률·상태 칩·파일 정보를 구조화한 아래쪽 상태 표시줄을 만든다."""
        status_bar = ttk.Frame(main)
        status_bar.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(self.px(8), 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            status_bar,
            orient="horizontal",
            length=self.px(150),
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
            style="Slim.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(side="left")

        self.progress_label = ttk.Label(status_bar, text="완료 0/0", style="Value.TLabel")
        self.progress_label.pack(side="left", padx=(self.px(8), self.px(12)))

        self.state_chip = self.make_chip(status_bar)
        self.state_chip.pack(side="left", padx=(0, self.px(6)))
        self.saved_chip = self.make_chip(status_bar)
        self.saved_chip.pack(side="left", padx=(0, self.px(12)))

        self.file_label = ttk.Label(status_bar, text="", style="Value.TLabel")
        self.file_label.pack(side="left")
        self.position_label = ttk.Label(status_bar, text="", style="Muted.TLabel")
        self.position_label.pack(side="left", padx=(self.px(8), 0))
        self.bbox_count_label = ttk.Label(status_bar, text="", style="Muted.TLabel")
        self.bbox_count_label.pack(side="left", padx=(self.px(8), 0))
        self.zoom_label = ttk.Label(status_bar, text="", style="Muted.TLabel")
        self.zoom_label.pack(side="left", padx=(self.px(8), 0))

        # 일시적인 안내 메시지(폴더 감시 중, 설정 적용됨 등)는 오른쪽 끝에.
        self.status_label = ttk.Label(status_bar, text="", style="Muted.TLabel")
        self.status_label.pack(side="right")

    def make_chip(self, parent: ttk.Frame) -> tk.Label:
        """상태를 색으로 보여주는 작은 칩 라벨을 만든다."""
        return tk.Label(
            parent,
            text="",
            bg=UI["panel"],
            fg=UI["text"],
            font=self.chip_font,
            padx=self.px(8),
            pady=self.px(1),
        )

    def set_chip(self, chip: tk.Label, text: str, color: str | None) -> None:
        """칩의 문구와 배경색을 바꾼다. color가 없으면 숨긴 것처럼 보인다."""
        if not text or color is None:
            chip.config(text="", bg=UI["panel"])
            return
        chip.config(text=text, bg=color, fg=contrast_text_color(color))

    # ------------------------------------------------------------------ 단축키

    def bind_shortcuts(self) -> None:
        """검수 속도를 위해 자주 쓰는 동작을 키보드에 연결한다."""
        bindings = {
            "<Left>": (self.previous_image, True),
            "<Right>": (self.next_image, True),
            "<a>": (self.previous_image, False),
            "<A>": (self.previous_image, False),
            "<d>": (self.next_image, False),
            "<D>": (self.next_image, False),
            "<Return>": (self.complete_and_go_next, False),
            "<Control-s>": (self.save_reviewed, False),
            "<n>": (self.add_default_bbox, False),
            "<N>": (self.add_default_bbox, False),
            "<Delete>": (self.delete_selected_bbox, False),
            "<BackSpace>": (self.delete_selected_bbox, False),
            "<Control-z>": (self.undo_bbox_change, False),
            "<Control-y>": (self.redo_bbox_change, False),
            "<f>": (self.fit_image_to_view, False),
            "<F>": (self.fit_image_to_view, False),
        }
        for sequence, (command, skip_in_listbox) in bindings.items():
            self.root.bind(
                sequence, self.guarded_shortcut(command, skip_in_listbox=skip_in_listbox)
            )

        for digit in range(10):
            self.root.bind(
                f"<Key-{digit}>",
                self.guarded_shortcut(self.select_class_by_digit, digit),
            )

    def guarded_shortcut(self, command, *arguments, skip_in_listbox: bool = False):
        """글자 입력 위젯(또는 목록)에 포커스가 있을 때 단축키가 끼어들지 않게 한다."""

        def handler(_event: tk.Event | None = None) -> str | None:
            focused_widget = self.focused_widget()
            if isinstance(focused_widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
                return None
            if skip_in_listbox and isinstance(focused_widget, tk.Listbox):
                # 방향키는 목록 안에서 목록 탐색으로 쓰인다.
                return None
            command(*arguments)
            return "break"

        return handler

    def focused_widget(self):
        try:
            return self.root.focus_get()
        except (KeyError, tk.TclError):
            return None

    def on_class_selected(self, _event: tk.Event | None = None) -> None:
        """class를 고르면 색 견본을 갱신하고 포커스를 사진으로 되돌린다."""
        self.update_class_swatch()
        self.canvas.focus_set()

    def update_class_swatch(self) -> None:
        """콤보 옆 색 견본을 현재 class 색으로 칠한다."""
        self.class_swatch.delete("all")
        size = self.px(14)
        self.class_swatch.create_rectangle(
            1, 1, size - 1, size - 1,
            fill=class_bbox_color(self.current_class_id()),
            outline=UI["line"],
        )

    def select_class_by_digit(self, digit: int) -> None:
        """숫자 키로 class를 고르고, 선택된 bbox가 있으면 그 class까지 바꾼다."""
        class_index = 9 if digit == 0 else digit - 1
        if class_index >= len(self.classes):
            return
        self.class_var.set(self.classes[class_index])
        self.update_class_swatch()
        if self.selected_bbox_index is not None:
            self.change_selected_class()
        else:
            self.update_status()

    # ------------------------------------------------------------------ 이미지 로드

    def load_current_image(self) -> None:
        self.status_label.config(text="")
        if not self.image_paths:
            self.current_image_path = None
            self.original_image = None
            self.image_load_failed = False
            self.bboxes = []
            self.has_unsaved_changes = False
            self.bbox_history.reset()
            self.selected_bbox_index = None
            self.render_empty_canvas(f"이미지가 없습니다\n{self.source_dir}")
            self.refresh_bbox_list()
            self.progress_var.set(0.0)
            self.progress_label.config(text="완료 0/0")
            self.set_chip(self.state_chip, "", None)
            self.set_chip(self.saved_chip, "", None)
            self.file_label.config(text="")
            self.position_label.config(text="")
            self.bbox_count_label.config(text="")
            self.zoom_label.config(text="")
            return

        self.current_image_index = min(
            max(0, self.current_image_index), len(self.image_paths) - 1
        )
        self.current_image_path = self.image_paths[self.current_image_index]
        self.bbox_history.reset()
        self.edit_original_bboxes = None
        self.has_unsaved_changes = False
        self.selected_bbox_index = None

        try:
            image = read_image(self.current_image_path)
        except Exception as exc:
            self.handle_image_load_failure(exc)
            return

        self.image_load_failed = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image_height, self.current_image_width = image.shape[:2]
        self.original_image = image
        self.reset_view_to_fit()

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
            self.bboxes = load_bboxes(label_path)
        except Exception as exc:
            messagebox.showerror(
                "자동 라벨 생성 오류",
                (
                    "이 이미지는 아직 자동 txt가 없습니다. 그래서 YOLO 모델로 txt를 새로 만들려고 했지만, "
                    "자동 생성 준비 또는 추론 단계에서 실패했습니다.\n\n"
                    "확인할 것:\n"
                    "1. GUI의 '모델 설정'에서 사용할 모델이 체크되어 있는지\n"
                    "2. 체크된 모델의 .pt 파일 경로가 실제 파일인지\n"
                    "3. 해당 이미지 파일을 OpenCV/YOLO가 읽을 수 있는지\n\n"
                    "실제 오류:\n"
                    f"{exc}"
                ),
            )
            self.bboxes = []

        self.render_image()
        self.update_status()

    def render_empty_canvas(self, message: str) -> None:
        """사진이 없을 때 안내 문구를 캔버스 가운데에 그린다."""
        self.canvas.delete("all")
        viewport_width, viewport_height = self.viewport_size()
        self.canvas.create_text(
            viewport_width / 2,
            viewport_height / 2,
            text=message,
            fill=UI["muted"],
            font=("Malgun Gothic", 11),
            justify="center",
        )
        self.update_scrollbars()

    def handle_image_load_failure(self, exc: Exception) -> None:
        """이미지를 못 읽었을 때 이전 사진의 bbox가 남아 덮어써지지 않게 정리한다."""
        self.image_load_failed = True
        self.original_image = None
        self.current_image_width = 0
        self.current_image_height = 0
        self.bboxes = []
        self.has_unsaved_changes = False
        self.photo_image = None
        self.render_empty_canvas("이미지를 읽을 수 없습니다")
        self.refresh_bbox_list()
        messagebox.showerror(
            "이미지 오류",
            (
                f"이미지를 읽을 수 없습니다: {self.current_image_path}\n\n"
                "이 사진은 저장 대상에서 제외됩니다. 파일이 손상되었거나 "
                "지원하지 않는 형식일 수 있습니다.\n\n"
                f"실제 오류:\n{exc}"
            ),
        )
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
        self.status_label.config(text="")

    def reinfer_current_image(self) -> None:
        """모델이나 임계값을 바꾼 뒤 현재 사진의 자동 라벨을 다시 만든다."""
        if self.current_image_path is None or self.image_load_failed:
            return
        if not messagebox.askyesno(
            "이 사진 재추론",
            "현재 화면의 bbox를 버리고 YOLO 모델로 다시 자동 생성합니다.\n계속할까요?",
        ):
            return

        auto_label_path = self.label_path_for(self.current_image_path, self.labels_auto_dir)
        try:
            self.create_auto_label_for_current_image(auto_label_path)
            self.bboxes = load_bboxes(auto_label_path)
        except Exception as exc:
            messagebox.showerror("재추론 오류", f"자동 라벨을 다시 만들지 못했습니다.\n\n실제 오류:\n{exc}")
            self.update_status()
            return

        self.bbox_history.reset()
        self.selected_bbox_index = None
        self.has_unsaved_changes = True
        self.render_bboxes()
        self.update_status()

    # ------------------------------------------------------------------ 설정 대화상자

    def choose_model_files(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("YOLO 모델 파일 선택")
        dialog.configure(bg=UI["panel"])
        dialog.transient(self.root)
        dialog.grab_set()

        model_path_vars: list[tk.StringVar] = []
        model_enabled_vars: list[tk.BooleanVar] = []
        model_entries = self.config.get("models", [])

        ttk.Label(dialog, text="사용").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Label(dialog, text="모델").grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(dialog, text=".pt 경로").grid(row=0, column=2, padx=6, pady=6, sticky="w")
        ttk.Label(dialog, text="상태").grid(row=0, column=4, padx=6, pady=6, sticky="w")

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

            status_label = ttk.Label(dialog, text="")
            status_label.grid(row=row_index, column=4, padx=6, pady=4, sticky="w")
            self.bind_model_path_status(path_var, status_label)

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
            style="Accent.TButton",
            command=lambda: self.apply_model_files(
                dialog,
                model_path_vars,
                model_enabled_vars,
                save_config=True,
            ),
        ).grid(row=button_row, column=3, padx=6, pady=8, sticky="e")

    def bind_model_path_status(self, path_var: tk.StringVar, status_label: ttk.Label) -> None:
        """입력한 .pt 경로가 실제 파일인지 대화상자에서 바로 보여준다."""

        def refresh_status(*_arguments: object) -> None:
            path_text = path_var.get().strip()
            if not path_text:
                status_label.config(text="경로 없음", foreground=UI["state_unsaved"])
            elif self.resolve_path(path_text).exists():
                status_label.config(text="파일 있음", foreground=UI["state_done"])
            else:
                status_label.config(text="파일 없음", foreground=UI["state_unsaved"])

        path_var.trace_add("write", refresh_status)
        refresh_status()

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
        validated_updates: list[tuple[dict[str, Any], bool, str | None]] = []
        for model_config, path_var, enabled_var in zip(
            model_entries,
            model_path_vars,
            model_enabled_vars,
        ):
            if not isinstance(model_config, dict):
                raise ValueError("each model config must be a mapping")
            is_enabled = bool(enabled_var.get())
            if not is_enabled:
                validated_updates.append((model_config, False, None))
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
            validated_updates.append(
                (model_config, True, self.path_for_config(selected_path))
            )

        if enabled_count == 0:
            raise ValueError("at least one model must be enabled")

        for model_config, is_enabled, config_path_text in validated_updates:
            model_config["enabled"] = is_enabled
            if config_path_text is not None:
                model_config["path"] = config_path_text

    def choose_source_dir(self) -> None:
        selected_dir = filedialog.askdirectory(
            title="이미지 폴더 선택",
            initialdir=str(self.source_dir),
        )
        if not selected_dir:
            return
        self.switch_source_dir(Path(selected_dir), watch=False)

    def choose_watched_source_dir(self) -> None:
        selected_dir = filedialog.askdirectory(
            title="드론 사진이 들어오는 폴더 선택",
            initialdir=str(self.source_dir),
        )
        if not selected_dir:
            return
        if self.switch_source_dir(Path(selected_dir), watch=True):
            self.status_label.config(text=f"폴더 감시 중: {self.source_dir}")

    def switch_source_dir(self, source_dir: Path, watch: bool) -> bool:
        """이미지 폴더를 바꾸기 전에 저장하고, 검수 상태도 폴더별로 갈아끼운다."""
        if not self.save_current_reviewed_state():
            return False

        self.persist_current_source_status()
        self.source_dir = source_dir
        self.status_source_key = source_key_for(self.source_dir)
        self.completed_image_keys = set(
            self.status_by_source.get(self.status_source_key, set())
        )
        self.config["source_dir"] = self.path_for_config(self.source_dir)
        self.watch_source_dir = watch
        self.config["watch_source_dir"] = watch
        self.save_config()
        self.current_image_path = None
        self.current_image_index = 0
        self.reload_images()
        return True

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
        self.root.after(SOURCE_WATCH_INTERVAL_MS, self.poll_source_dir)

    def poll_source_dir(self) -> None:
        """감시 중인 폴더에 새 사진이 들어왔는지 주기적으로 확인한다."""
        try:
            if self.watch_source_dir:
                self.refresh_watched_images()
        except Exception as exc:
            self.watch_source_dir = False
            self.status_label.config(text=f"폴더 감시를 멈췄습니다: {exc}")
        finally:
            self.schedule_source_watch()

    def refresh_watched_images(self) -> None:
        """감시 폴더의 파일 목록 변화를 현재 목록에 반영한다."""
        image_paths = self.find_images(self.source_dir)
        image_path_set = set(image_paths)
        if image_path_set == self.known_image_paths:
            return

        new_image_count = len(image_path_set - self.known_image_paths)
        current_image_path = self.current_image_path
        self.image_paths = image_paths
        self.known_image_paths = image_path_set

        if current_image_path in self.image_paths:
            self.current_image_index = self.image_paths.index(current_image_path)
            self.update_status()
        else:
            self.current_image_index = 0
            self.load_current_image()

        if new_image_count > 0:
            self.status_label.config(
                text=f"새 사진 {new_image_count}장 감지됨 · 총 {len(self.image_paths)}장"
            )

    # ------------------------------------------------------------------ 뷰포트

    def viewport_size(self) -> tuple[float, float]:
        """캔버스에서 실제로 보이는 영역의 크기를 얻는다."""
        return (
            max(1.0, float(self.canvas.winfo_width())),
            max(1.0, float(self.canvas.winfo_height())),
        )

    def display_size(self) -> tuple[float, float]:
        """현재 배율에서 이미지 전체가 차지하는 가상 크기를 얻는다."""
        return (
            self.current_image_width * self.display_scale,
            self.current_image_height * self.display_scale,
        )

    def calculate_fit_scale(self) -> float:
        """이미지 전체가 뷰포트 안에 들어오는 배율을 계산한다."""
        if self.current_image_width <= 0 or self.current_image_height <= 0:
            return 1.0
        viewport_width, viewport_height = self.viewport_size()
        return min(
            viewport_width / self.current_image_width,
            viewport_height / self.current_image_height,
        )

    def reset_view_to_fit(self) -> None:
        """이미지를 뷰포트에 맞추고 화면 이동 위치를 처음으로 되돌린다."""
        self.fit_scale = self.calculate_fit_scale()
        self.display_scale = self.fit_scale
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0

    def fit_image_to_view(self) -> None:
        """F 키나 버튼으로 언제든 이미지를 화면에 다시 맞춘다."""
        if self.original_image is None:
            return
        self.reset_view_to_fit()
        self.render_image()
        self.update_status()

    def clamp_view_offsets(self) -> None:
        """화면 이동 위치를 이미지 밖으로 나가지 않게 제한한다."""
        viewport_width, viewport_height = self.viewport_size()
        display_width, display_height = self.display_size()
        self.view_offset_x = min(
            max(0.0, self.view_offset_x), max(0.0, display_width - viewport_width)
        )
        self.view_offset_y = min(
            max(0.0, self.view_offset_y), max(0.0, display_height - viewport_height)
        )

    def on_canvas_configure(self, _event: tk.Event) -> None:
        """창 크기가 바뀌면 화면에 맞춘 배율과 이동 범위를 다시 계산한다."""
        if self.canvas_resize_job is not None:
            self.root.after_cancel(self.canvas_resize_job)
        self.canvas_resize_job = self.root.after(
            CANVAS_RESIZE_DEBOUNCE_MS, self.apply_canvas_resize
        )

    def apply_canvas_resize(self) -> None:
        self.canvas_resize_job = None
        if self.original_image is None:
            if not self.image_paths:
                self.render_empty_canvas(f"이미지가 없습니다\n{self.source_dir}")
            return

        was_fitted = math.isclose(self.display_scale, self.fit_scale, rel_tol=1e-6)
        self.fit_scale = self.calculate_fit_scale()
        if was_fitted:
            self.display_scale = self.fit_scale
            self.view_offset_x = 0.0
            self.view_offset_y = 0.0
        self.render_image()
        self.update_status()

    def render_image(self) -> None:
        """보이는 영역만 잘라 배율에 맞춰 그린다.

        이미지 전체를 배율만큼 키우면 4배 확대에서 수백 MB 배열이 만들어지므로,
        화면에 실제로 보이는 부분만 만들어 확대 비용을 일정하게 유지한다.
        """
        if self.original_image is None:
            self.canvas.delete("all")
            self.update_scrollbars()
            return

        self.clamp_view_offsets()
        viewport_width, viewport_height = self.viewport_size()
        display_width, display_height = self.display_size()

        self.image_origin_x = max(0.0, (viewport_width - display_width) / 2)
        self.image_origin_y = max(0.0, (viewport_height - display_height) / 2)

        source_left = int(self.view_offset_x / self.display_scale)
        source_top = int(self.view_offset_y / self.display_scale)
        source_right = min(
            self.current_image_width,
            int(math.ceil((self.view_offset_x + viewport_width) / self.display_scale)),
        )
        source_bottom = min(
            self.current_image_height,
            int(math.ceil((self.view_offset_y + viewport_height) / self.display_scale)),
        )
        source_right = max(source_right, source_left + 1)
        source_bottom = max(source_bottom, source_top + 1)

        visible_image = self.original_image[
            source_top:source_bottom, source_left:source_right
        ]
        target_width = max(1, int(round((source_right - source_left) * self.display_scale)))
        target_height = max(1, int(round((source_bottom - source_top) * self.display_scale)))
        interpolation = self.resize_interpolation()
        visible_image = cv2.resize(
            visible_image,
            (target_width, target_height),
            interpolation=interpolation,
        )

        self.photo_image = self.cv_image_to_photo_image(visible_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            source_left * self.display_scale - self.view_offset_x + self.image_origin_x,
            source_top * self.display_scale - self.view_offset_y + self.image_origin_y,
            anchor="nw",
            image=self.photo_image,
        )
        self.update_scrollbars()
        self.render_bboxes()

    def resize_interpolation(self) -> int:
        """축소는 부드럽게, 크게 확대할 때는 픽셀 경계가 보이게 고른다."""
        if self.display_scale < 1.0:
            return cv2.INTER_AREA
        if self.display_scale < 3.0:
            return cv2.INTER_LINEAR
        return cv2.INTER_NEAREST

    def update_scrollbars(self) -> None:
        """현재 보이는 범위를 스크롤바 위치에 반영한다."""
        viewport_width, viewport_height = self.viewport_size()
        display_width, display_height = self.display_size()
        if display_width <= 0 or display_height <= 0:
            self.horizontal_scrollbar.set(0.0, 1.0)
            self.vertical_scrollbar.set(0.0, 1.0)
            return

        self.horizontal_scrollbar.set(
            self.view_offset_x / display_width,
            min(1.0, (self.view_offset_x + viewport_width) / display_width),
        )
        self.vertical_scrollbar.set(
            self.view_offset_y / display_height,
            min(1.0, (self.view_offset_y + viewport_height) / display_height),
        )

    def on_horizontal_scroll(self, command: str, value: str, unit: str = "") -> None:
        display_width, _display_height = self.display_size()
        viewport_width, _viewport_height = self.viewport_size()
        if command == "moveto":
            self.view_offset_x = float(value) * display_width
        elif command == "scroll":
            step = viewport_width if unit == "pages" else viewport_width / 10
            self.view_offset_x += float(value) * step
        self.render_image()

    def on_vertical_scroll(self, command: str, value: str, unit: str = "") -> None:
        _display_width, display_height = self.display_size()
        _viewport_width, viewport_height = self.viewport_size()
        if command == "moveto":
            self.view_offset_y = float(value) * display_height
        elif command == "scroll":
            step = viewport_height if unit == "pages" else viewport_height / 10
            self.view_offset_y += float(value) * step
        self.render_image()

    def image_to_canvas(self, image_x: float, image_y: float) -> tuple[float, float]:
        """이미지 좌표를 화면 좌표로 바꾼다."""
        return (
            image_x * self.display_scale - self.view_offset_x + self.image_origin_x,
            image_y * self.display_scale - self.view_offset_y + self.image_origin_y,
        )

    def canvas_to_image(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        """화면 좌표를 이미지 좌표로 바꾼다."""
        return (
            (canvas_x - self.image_origin_x + self.view_offset_x) / self.display_scale,
            (canvas_y - self.image_origin_y + self.view_offset_y) / self.display_scale,
        )

    def bbox_canvas_rect(self, bbox: BBox) -> tuple[float, float, float, float]:
        """bbox를 현재 배율·이동 상태의 화면 사각형으로 바꾼다."""
        left, top, right, bottom = bbox_to_canvas_rect(
            bbox,
            self.current_image_width,
            self.current_image_height,
        )
        canvas_left, canvas_top = self.image_to_canvas(left, top)
        canvas_right, canvas_bottom = self.image_to_canvas(right, bottom)
        return canvas_left, canvas_top, canvas_right, canvas_bottom

    # ------------------------------------------------------------------ bbox 그리기

    def render_bboxes(self) -> None:
        self.canvas.delete("bbox")
        for index, bbox in enumerate(self.bboxes):
            self.draw_bbox(index, bbox)
        self.refresh_bbox_list()

    def draw_bbox(self, index: int, bbox: BBox) -> None:
        """bbox 사각형과 class 이름 칩을 그린다."""
        left, top, right, bottom = self.bbox_canvas_rect(bbox)
        color = class_bbox_color(bbox.class_id)
        is_selected = index == self.selected_bbox_index

        self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            outline=color,
            width=self.px(3) if is_selected else self.px(2),
            tags=("bbox",),
        )

        # 복잡한 사진 위에서도 읽히도록 class 이름은 색 채운 칩으로 그린다.
        class_name = self.class_name_for(bbox.class_id)
        text_width = self.canvas_chip_font.measure(class_name)
        text_height = self.canvas_chip_font.metrics("linespace")
        pad_x = self.px(5)
        chip_height = text_height + self.px(3)
        chip_top = top - chip_height - self.px(2)
        if chip_top < 2:
            chip_top = top + self.px(2)
        self.canvas.create_rectangle(
            left,
            chip_top,
            left + text_width + pad_x * 2,
            chip_top + chip_height,
            fill=color,
            outline="#ffffff" if is_selected else color,
            width=1,
            tags=("bbox",),
        )
        self.canvas.create_text(
            left + pad_x,
            chip_top + chip_height / 2,
            anchor="w",
            text=class_name,
            fill=contrast_text_color(color),
            font=self.canvas_chip_font,
            tags=("bbox",),
        )

        if is_selected:
            self.render_resize_handles(left, top, right, bottom)

    def refresh_bbox_list(self) -> None:
        """오른쪽 bbox 목록을 class 색과 함께 다시 채운다."""
        self.updating_bbox_list = True
        self.bbox_list.delete(0, tk.END)
        for index, bbox in enumerate(self.bboxes):
            self.bbox_list.insert(tk.END, f" {index}  ·  {self.class_name_for(bbox.class_id)}")
            self.bbox_list.itemconfig(index, foreground=class_bbox_color(bbox.class_id))
        if self.selected_bbox_index is not None and 0 <= self.selected_bbox_index < len(self.bboxes):
            self.bbox_list.selection_set(self.selected_bbox_index)
            self.bbox_list.see(self.selected_bbox_index)
        self.bbox_list_title.config(text=f"bbox 목록 · {len(self.bboxes)}개")
        self.updating_bbox_list = False

    def class_name_for(self, class_id: int) -> str:
        """class id를 사람이 읽을 이름으로 바꾼다."""
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"알 수 없는 class {class_id}"

    def render_resize_handles(self, left: float, top: float, right: float, bottom: float) -> None:
        """선택된 bbox의 크기를 조절할 수 있는 화면 핸들을 그린다."""
        half_size = self.px(HANDLE_SIZE) / 2
        for handle_x, handle_y in self.resize_handle_points(left, top, right, bottom).values():
            self.canvas.create_rectangle(
                handle_x - half_size,
                handle_y - half_size,
                handle_x + half_size,
                handle_y + half_size,
                outline=UI["canvas"],
                fill="#ffcc00",
                width=1,
                tags=("bbox", "bbox_handle"),
            )

    # ------------------------------------------------------------------ 마우스

    def on_mouse_wheel(self, event: tk.Event) -> None:
        """마우스 위치를 유지하면서 휠 한 단계마다 확대·축소한다."""
        if self.original_image is None:
            return

        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            zoom_factor = ZOOM_OUT_FACTOR
        else:
            zoom_factor = ZOOM_IN_FACTOR

        minimum_scale = max(self.fit_scale * 0.25, 0.02)
        next_scale = max(
            minimum_scale, min(self.display_scale * zoom_factor, MAXIMUM_ZOOM_SCALE)
        )
        if math.isclose(next_scale, self.display_scale, rel_tol=1e-9):
            return

        image_x, image_y = self.canvas_to_image(event.x, event.y)
        self.display_scale = next_scale

        viewport_width, viewport_height = self.viewport_size()
        display_width, display_height = self.display_size()
        next_origin_x = max(0.0, (viewport_width - display_width) / 2)
        next_origin_y = max(0.0, (viewport_height - display_height) / 2)
        self.view_offset_x = image_x * self.display_scale - (event.x - next_origin_x)
        self.view_offset_y = image_y * self.display_scale - (event.y - next_origin_y)

        self.render_image()
        self.update_status()

    def on_pan_start(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y, self.view_offset_x, self.view_offset_y)
        self.canvas.config(cursor="fleur")

    def on_pan_drag(self, event: tk.Event) -> None:
        if self.pan_start is None:
            return
        start_x, start_y, start_offset_x, start_offset_y = self.pan_start
        self.view_offset_x = start_offset_x - (event.x - start_x)
        self.view_offset_y = start_offset_y - (event.y - start_y)
        self.render_image()

    def on_pan_end(self, _event: tk.Event) -> None:
        self.pan_start = None
        self.canvas.config(cursor=self.hover_cursor)

    def on_canvas_motion(self, event: tk.Event) -> None:
        """마우스가 무엇 위에 있는지 커서 모양으로 미리 알려준다."""
        if self.edit_mode is not None or self.pan_start is not None:
            return
        if self.original_image is None:
            cursor = ""
        else:
            handle_name = self.find_resize_handle_at_canvas_point(event.x, event.y)
            if handle_name is not None:
                cursor = RESIZE_CURSORS[handle_name]
            elif self.find_bbox_at_canvas_point(event.x, event.y) is not None:
                cursor = "fleur"
            else:
                cursor = "crosshair"
        if cursor != self.hover_cursor:
            self.hover_cursor = cursor
            try:
                self.canvas.config(cursor=cursor)
            except tk.TclError:
                self.hover_cursor = ""
                self.canvas.config(cursor="")

    def on_mouse_down(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self.original_image is None:
            return

        handle_name = self.find_resize_handle_at_canvas_point(event.x, event.y)
        if handle_name is not None:
            self.start_bbox_edit("resize", event.x, event.y, handle_name)
            return

        clicked_bbox_index = self.find_bbox_at_canvas_point(event.x, event.y)
        if clicked_bbox_index is not None:
            self.selected_bbox_index = clicked_bbox_index
            self.render_bboxes()
            self.start_bbox_edit("move", event.x, event.y, None)
            return

        self.selected_bbox_index = None
        self.clear_bbox_edit()
        self.render_bboxes()
        self.edit_mode = "create"
        self.new_bbox_start = (event.x, event.y)

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.edit_mode == "create":
            self.render_new_bbox_preview(event.x, event.y)
            return
        if self.selected_bbox_index is None or self.edit_mode is None:
            return

        if self.edit_mode == "move":
            self.move_selected_bbox(event.x, event.y)
        elif self.edit_mode == "resize":
            self.resize_selected_bbox(event.x, event.y)

    def on_mouse_up(self, event: tk.Event) -> None:
        if self.edit_mode is None:
            return

        completed_edit_mode = self.edit_mode
        previous_bboxes = self.edit_original_bboxes
        new_bbox_start = self.new_bbox_start
        self.clear_bbox_edit()

        if completed_edit_mode == "create":
            self.finish_new_bbox(new_bbox_start, event.x, event.y)
        elif (
            completed_edit_mode in {"move", "resize"}
            and previous_bboxes is not None
            and previous_bboxes != self.bboxes
        ):
            self.bbox_history.record(previous_bboxes)
            self.mark_current_reviewing()

        self.edit_original_bboxes = None
        self.update_status()

    def render_new_bbox_preview(self, canvas_x: float, canvas_y: float) -> None:
        """드래그하는 동안 새 bbox를 점선·십자선·실측 크기로 미리 보여준다."""
        if self.new_bbox_start is None:
            return
        start_x, start_y = self.new_bbox_start
        color = class_bbox_color(self.current_class_id())
        viewport_width, viewport_height = self.viewport_size()

        self.canvas.delete("new_bbox")
        self.canvas.create_line(
            0, canvas_y, viewport_width, canvas_y,
            fill=UI["muted"], dash=(2, 4), tags=("new_bbox",),
        )
        self.canvas.create_line(
            canvas_x, 0, canvas_x, viewport_height,
            fill=UI["muted"], dash=(2, 4), tags=("new_bbox",),
        )
        self.canvas.create_rectangle(
            start_x,
            start_y,
            canvas_x,
            canvas_y,
            outline=color,
            dash=(5, 3),
            width=self.px(2),
            tags=("new_bbox",),
        )

        # 이미지 픽셀 기준 실제 크기를 함께 보여준다.
        start_image_x, start_image_y = self.canvas_to_image(start_x, start_y)
        end_image_x, end_image_y = self.canvas_to_image(canvas_x, canvas_y)
        size_text = (
            f"{abs(end_image_x - start_image_x):.0f} × "
            f"{abs(end_image_y - start_image_y):.0f} px"
        )
        self.canvas.create_text(
            canvas_x + self.px(12),
            canvas_y + self.px(12),
            anchor="nw",
            text=size_text,
            fill=color,
            font=self.canvas_chip_font,
            tags=("new_bbox",),
        )

    def finish_new_bbox(
        self,
        new_bbox_start: tuple[float, float] | None,
        canvas_x: float,
        canvas_y: float,
    ) -> None:
        """빈 곳을 드래그해 그린 사각형을 실제 bbox로 만든다."""
        self.canvas.delete("new_bbox")
        if new_bbox_start is None:
            return

        start_x, start_y = new_bbox_start
        minimum_drag = self.px(MIN_NEW_BBOX_DRAG_PIXELS)
        if (
            abs(canvas_x - start_x) < minimum_drag
            or abs(canvas_y - start_y) < minimum_drag
        ):
            self.render_bboxes()
            return

        start_image_x, start_image_y = self.canvas_to_image(start_x, start_y)
        end_image_x, end_image_y = self.canvas_to_image(canvas_x, canvas_y)
        bbox = bbox_from_canvas_rect(
            self.current_class_id(),
            start_image_x,
            start_image_y,
            end_image_x,
            end_image_y,
            self.current_image_width,
            self.current_image_height,
        )

        self.bbox_history.record(self.bboxes)
        self.bboxes = create_bbox(self.bboxes, bbox)
        self.mark_current_reviewing()
        self.selected_bbox_index = len(self.bboxes) - 1
        self.render_bboxes()

    def add_default_bbox(self) -> None:
        """선택된 class로 이미지 중앙에 기본 크기 bbox를 추가한다."""
        if self.current_image_path is None or self.original_image is None:
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
        self.new_bbox_start = None

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
        image_x, image_y = self.canvas_to_image(canvas_x, canvas_y)
        pointer_x = max(0.0, min(image_x, float(self.current_image_width)))
        pointer_y = max(0.0, min(image_y, float(self.current_image_height)))
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
        bbox_index = self.find_bbox_at_canvas_point(event.x, event.y)
        if bbox_index is None:
            return
        self.selected_bbox_index = bbox_index
        self.delete_selected_bbox()

    def find_resize_handle_at_canvas_point(self, canvas_x: float, canvas_y: float) -> str | None:
        if self.selected_bbox_index is None or self.selected_bbox_index >= len(self.bboxes):
            return None

        left, top, right, bottom = self.bbox_canvas_rect(
            self.bboxes[self.selected_bbox_index]
        )
        handle_points = self.resize_handle_points(left, top, right, bottom)
        tolerance = self.px(HANDLE_SIZE)
        for handle_name in RESIZE_HANDLE_NAMES:
            handle_x, handle_y = handle_points[handle_name]
            if abs(canvas_x - handle_x) <= tolerance and abs(canvas_y - handle_y) <= tolerance:
                return handle_name
        return None

    def find_bbox_at_canvas_point(self, canvas_x: float, canvas_y: float) -> int | None:
        for index in range(len(self.bboxes) - 1, -1, -1):
            left, top, right, bottom = self.bbox_canvas_rect(self.bboxes[index])
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

    # ------------------------------------------------------------------ 편집 명령

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
        self.update_class_swatch()
        self.save_classes()
        self.save_training_classes()
        self.save_training_dataset_config()
        self.save_config()
        self.status_label.config(text=f"class 추가됨: {class_name}")

    # ------------------------------------------------------------------ 저장·검수 상태

    def save_reviewed(self) -> None:
        if self.save_current_reviewed_state(force=True):
            self.update_status(saved=True)

    def complete_and_go_next(self) -> None:
        """Enter 한 번으로 검수 완료 표시 후 다음 미검수 사진으로 넘어간다."""
        if self.current_image_path is None or self.image_load_failed:
            return
        image_key = self.current_image_key()
        if image_key not in self.completed_image_keys:
            if not self.mark_current_complete():
                return
        self.next_unreviewed_image()

    def toggle_review_complete(self) -> None:
        """현재 이미지를 검수 완료로 표시하거나 다시 검수 중으로 되돌린다."""
        if self.current_image_path is None or self.image_load_failed:
            return

        image_key = self.current_image_key()
        if image_key in self.completed_image_keys:
            self.completed_image_keys.remove(image_key)
            self.remove_training_pair()
            if self.save_review_status():
                self.update_status()
            return

        self.mark_current_complete()

    def mark_current_complete(self) -> bool:
        """현재 이미지를 검수 완료로 표시하고 학습 폴더로 내보낸다."""
        if not self.save_current_reviewed_state(force=True):
            return False
        try:
            self.export_current_training_pair()
        except Exception as exc:
            messagebox.showerror(
                "학습 폴더 저장 오류",
                f"학습 폴더로 내보내지 못했습니다.\n\n실제 오류:\n{exc}",
            )
            return False

        self.completed_image_keys.add(self.current_image_key())
        if not self.save_review_status():
            return False
        self.update_status(saved=True)
        return True

    def mark_current_reviewing(self) -> None:
        """편집이 생기면 저장 필요 표시를 켜고, 완료 표시는 해제한다."""
        if self.current_image_path is None:
            return
        self.has_unsaved_changes = True
        image_key = self.current_image_key()
        if image_key not in self.completed_image_keys:
            return
        self.completed_image_keys.remove(image_key)
        self.save_review_status()

    def persist_current_source_status(self) -> None:
        """메모리의 현재 폴더 완료 목록을 전체 상태 표에 반영한다."""
        self.status_by_source[self.status_source_key] = set(self.completed_image_keys)

    def save_review_status(self) -> bool:
        """메모리의 검수 완료 목록을 review_status.json에 저장한다."""
        self.persist_current_source_status()
        try:
            save_status_by_source(self.review_status_path, self.status_by_source)
        except Exception as exc:
            messagebox.showerror(
                "검수 상태 저장 오류",
                f"검수 완료 상태를 저장하지 못했습니다.\n\n실제 오류:\n{exc}",
            )
            return False
        return True

    def save_current_reviewed_state(self, force: bool = False) -> bool:
        """현재 이미지의 bbox 목록을 검수 txt로 저장한다.

        고친 것이 없으면 파일을 만들지 않는다. 그래야 사진을 넘겨보기만 해도
        전부 '검수 중'으로 바뀌는 오해가 생기지 않는다.
        """
        if self.current_image_path is None or self.image_load_failed:
            return True
        if not force and not self.has_unsaved_changes:
            return True

        self.clear_bbox_edit()
        try:
            reviewed_label_path = self.label_path_for(
                self.current_image_path,
                self.labels_reviewed_dir,
            )
            save_bboxes(reviewed_label_path, self.bboxes)
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

        self.has_unsaved_changes = False
        return True

    def export_current_training_pair(self) -> None:
        """검수 완료된 사진과 라벨만 학습 폴더로 복사한다."""
        if self.current_image_path is None or self.dataset_output_dir is None:
            return

        image_output_path, label_output_path = self.training_pair_paths(
            self.current_image_path
        )
        image_output_path.parent.mkdir(parents=True, exist_ok=True)
        label_output_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self.current_image_path, image_output_path)
        save_bboxes(label_output_path, self.bboxes)
        self.save_training_classes()
        self.save_training_dataset_config()

    def remove_training_pair(self) -> None:
        """검수 완료를 취소하면 학습 폴더의 복사본도 함께 뺀다."""
        if self.current_image_path is None or self.dataset_output_dir is None:
            return

        for output_path in self.training_pair_paths(self.current_image_path):
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                continue

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
        self.save_training_dataset_config()

    def save_training_classes(self) -> None:
        if self.dataset_output_dir is None:
            return
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_output_dir / "classes.txt").write_text(
            "\n".join(self.classes) + "\n",
            encoding="utf-8",
        )

    def save_training_dataset_config(self) -> None:
        """YOLO 학습에 바로 쓸 수 있게 data.yaml을 만들어 둔다."""
        if self.dataset_output_dir is None:
            return
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)
        dataset_config = {
            "path": str(self.dataset_output_dir),
            "train": "images/train",
            "val": "images/train",
            "names": {index: name for index, name in enumerate(self.classes)},
        }
        (self.dataset_output_dir / "data.yaml").write_text(
            yaml.safe_dump(dataset_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ 탐색

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
        current_index = min(max(0, self.current_image_index), len(image_keys) - 1)
        next_image_index = find_next_uncompleted_index(
            image_keys,
            self.completed_image_keys,
            current_index,
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
            self.update_status()
            return

        self.current_image_index = next_image_index
        self.load_current_image()

    def current_class_id(self) -> int:
        class_name = self.class_var.get() if hasattr(self, "class_var") else ""
        if class_name not in self.classes:
            return 0
        return self.classes.index(class_name)

    # ------------------------------------------------------------------ 상태 표시

    def update_status(self, saved: bool = False) -> None:
        if self.current_image_path is None:
            return

        completed_count = sum(
            1
            for image_path in self.image_paths
            if self.image_key_for(image_path) in self.completed_image_keys
        )
        self.progress_var.set(
            completed_count * 100.0 / max(1, len(self.image_paths))
        )
        self.progress_label.config(
            text=f"완료 {completed_count}/{len(self.image_paths)}"
        )
        self.file_label.config(text=self.current_image_path.name)
        self.position_label.config(
            text=f"{self.current_image_index + 1}/{len(self.image_paths)}"
        )

        if self.image_load_failed:
            self.set_chip(self.state_chip, "읽기 실패", UI["state_unsaved"])
            self.set_chip(self.saved_chip, "", None)
            self.bbox_count_label.config(text="")
            self.zoom_label.config(text="")
            return

        image_key = self.current_image_key()
        reviewed_label_path = self.label_path_for(
            self.current_image_path,
            self.labels_reviewed_dir,
        )
        if image_key in self.completed_image_keys:
            self.set_chip(self.state_chip, "검수 완료", UI["state_done"])
            self.review_complete_button.config(text="검수 재개")
        elif reviewed_label_path.exists():
            self.set_chip(self.state_chip, "검수 중", UI["state_reviewing"])
            self.review_complete_button.config(text="검수 완료")
        else:
            self.set_chip(self.state_chip, "자동 생성", UI["state_auto"])
            self.review_complete_button.config(text="검수 완료")

        if saved:
            self.set_chip(self.saved_chip, "저장됨", UI["state_done"])
        elif self.has_unsaved_changes:
            self.set_chip(self.saved_chip, "저장 안 됨", UI["state_unsaved"])
        else:
            self.set_chip(self.saved_chip, "", None)

        self.bbox_count_label.config(text=f"bbox {len(self.bboxes)}개")
        self.zoom_label.config(text=f"{self.display_scale * 100:.0f}%")

    def on_close(self) -> None:
        """창을 닫기 전에 저장하지 않은 변경을 확인한다."""
        if self.has_unsaved_changes:
            answer = messagebox.askyesnocancel(
                "저장하지 않은 변경",
                "현재 사진에 저장하지 않은 변경이 있습니다.\n저장하고 닫을까요?",
            )
            if answer is None:
                return
            if answer and not self.save_current_reviewed_state(force=True):
                return
        self.root.destroy()

    # ------------------------------------------------------------------ 경로·설정

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
        if not source_dir.is_dir():
            raise NotADirectoryError(f"source_dir is not a directory: {source_dir}")
        return sorted(
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def cv_image_to_photo_image(image) -> tk.PhotoImage:
        """화면에 뿌릴 이미지를 Tkinter가 읽을 수 있는 형태로 바꾼다.

        압축을 끄고 base64를 거치지 않으면 같은 화면을 절반 시간에 그린다.
        """
        success, encoded_image = cv2.imencode(
            ".png",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        )
        if not success:
            raise ValueError("failed to encode image for Tkinter display")
        return tk.PhotoImage(data=encoded_image.tobytes(), format="PNG")


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
    enable_windows_dpi_awareness()
    root = tk.Tk()
    try:
        ReviewApp(root, args.config.resolve())
    except Exception as exc:
        root.withdraw()
        messagebox.showerror(
            "검수 GUI를 시작하지 못했습니다",
            (
                "설정을 읽는 중 문제가 생겨 GUI를 열 수 없습니다.\n\n"
                "확인할 것:\n"
                "1. autolabel_config.yaml의 source_dir 폴더가 실제로 있는지\n"
                "2. classes_path(classes.txt)가 있는지\n"
                "3. models 항목의 conf/iou/class_map이 채워져 있는지\n\n"
                f"실제 오류:\n{type(exc).__name__}: {exc}"
            ),
        )
        root.destroy()
        raise SystemExit(1) from exc
    root.mainloop()


if __name__ == "__main__":
    main()
