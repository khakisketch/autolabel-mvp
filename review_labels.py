from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class BBox:
    """검수 단계에서 다루는 YOLO bbox 1개."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class BBoxHistory:
    """한 이미지의 bbox 목록 변경을 실행 취소·다시 실행할 수 있게 보관한다."""

    def __init__(self) -> None:
        self.undo_states: list[list[BBox]] = []
        self.redo_states: list[list[BBox]] = []

    def reset(self) -> None:
        """이미지가 바뀔 때 이전 이미지의 변경 이력을 비운다."""
        self.undo_states.clear()
        self.redo_states.clear()

    def record(self, previous_bboxes: list[BBox]) -> None:
        """변경 직전 bbox 목록을 기록하고 기존 다시 실행 이력을 비운다."""
        self.undo_states.append(list(previous_bboxes))
        self.redo_states.clear()

    def undo(self, current_bboxes: list[BBox]) -> list[BBox] | None:
        """현재 bbox 목록을 다시 실행용으로 보관하고 직전 목록을 반환한다."""
        if not self.undo_states:
            return None
        self.redo_states.append(list(current_bboxes))
        return self.undo_states.pop()

    def redo(self, current_bboxes: list[BBox]) -> list[BBox] | None:
        """현재 bbox 목록을 실행 취소용으로 보관하고 취소했던 목록을 반환한다."""
        if not self.redo_states:
            return None
        self.undo_states.append(list(current_bboxes))
        return self.redo_states.pop()


def load_classes(classes_path: Path) -> list[str]:
    """classes.txt 파일에서 class 이름 목록을 읽는다."""
    if not classes_path.exists():
        raise FileNotFoundError(f"classes_path does not exist: {classes_path}")

    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_yolo_label_line(label_line: str) -> BBox:
    """YOLO txt 한 줄을 BBox로 변환한다."""
    parts = label_line.split()
    if len(parts) != 5:
        raise ValueError(f"invalid YOLO label line: {label_line}")

    class_id = int(parts[0])
    x_center, y_center, width, height = [float(value) for value in parts[1:]]
    return BBox(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def format_yolo_label_line(bbox: BBox) -> str:
    """BBox를 YOLO txt 한 줄로 변환한다."""
    return (
        f"{bbox.class_id} "
        f"{bbox.x_center:.6f} "
        f"{bbox.y_center:.6f} "
        f"{bbox.width:.6f} "
        f"{bbox.height:.6f}"
    )


def load_bboxes(label_path: Path) -> list[BBox]:
    """YOLO txt 파일에서 bbox 목록을 읽는다."""
    if not label_path.exists():
        return []

    label_lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [parse_yolo_label_line(label_line) for label_line in label_lines]


def resolve_label_for_review(
    reviewed_label_path: Path,
    auto_label_path: Path,
    create_auto_label: Callable[[], None],
) -> Path:
    """검수 txt를 우선하고, txt가 없으면 자동 라벨을 생성해 경로를 반환한다."""
    if reviewed_label_path.exists():
        return reviewed_label_path
    if not auto_label_path.exists():
        create_auto_label()
    if not auto_label_path.exists():
        raise RuntimeError(f"auto label was not created: {auto_label_path}")
    return auto_label_path


def save_bboxes(label_path: Path, bboxes: list[BBox]) -> None:
    """bbox 목록을 YOLO txt 파일로 저장한다."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_text = "\n".join(format_yolo_label_line(bbox) for bbox in bboxes)
    if label_text:
        label_text += "\n"
    label_path.write_text(label_text, encoding="utf-8")


def create_bbox(bboxes: list[BBox], bbox: BBox) -> list[BBox]:
    """검수중 bbox 목록에 새 bbox를 추가한다."""
    return [*bboxes, bbox]


def update_bbox(bboxes: list[BBox], index: int, bbox: BBox) -> list[BBox]:
    """검수중 bbox 목록에서 선택한 bbox를 교체한다."""
    validate_bbox_index(bboxes, index)
    updated_bboxes = list(bboxes)
    updated_bboxes[index] = bbox
    return updated_bboxes


def delete_bbox(bboxes: list[BBox], index: int) -> list[BBox]:
    """검수중 bbox 목록에서 선택한 bbox를 삭제한다."""
    validate_bbox_index(bboxes, index)
    return [bbox for bbox_index, bbox in enumerate(bboxes) if bbox_index != index]


def change_bbox_class(bboxes: list[BBox], index: int, class_id: int) -> list[BBox]:
    """검수중 bbox 목록에서 선택한 bbox의 class를 바꾼다."""
    validate_bbox_index(bboxes, index)
    return update_bbox(bboxes, index, replace(bboxes[index], class_id=class_id))


def bbox_from_canvas_rect(
    class_id: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
    image_width: int,
    image_height: int,
) -> BBox:
    """화면 좌표 사각형을 YOLO 정규화 bbox로 변환한다."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    left, right = sorted((clamp(left, 0, image_width), clamp(right, 0, image_width)))
    top, bottom = sorted((clamp(top, 0, image_height), clamp(bottom, 0, image_height)))

    width = (right - left) / image_width
    height = (bottom - top) / image_height
    x_center = ((left + right) / 2) / image_width
    y_center = ((top + bottom) / 2) / image_height

    return BBox(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def bbox_to_canvas_rect(
    bbox: BBox,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """YOLO 정규화 bbox를 화면 좌표 사각형으로 변환한다."""
    left = (bbox.x_center - bbox.width / 2) * image_width
    top = (bbox.y_center - bbox.height / 2) * image_height
    right = (bbox.x_center + bbox.width / 2) * image_width
    bottom = (bbox.y_center + bbox.height / 2) * image_height
    return left, top, right, bottom


def clamp(value: float, minimum: float, maximum: float) -> float:
    """값을 최소/최대 범위 안으로 제한한다."""
    return max(minimum, min(value, maximum))


def validate_bbox_index(bboxes: list[BBox], index: int) -> None:
    """bbox 목록의 선택 번호가 유효한지 확인한다."""
    if index < 0 or index >= len(bboxes):
        raise IndexError(f"bbox index out of range: {index}")
