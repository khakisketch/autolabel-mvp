from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from image_io import IMAGE_EXTENSIONS, read_image, write_image


__all__ = ["IMAGE_EXTENSIONS"]


@dataclass(frozen=True)
class ModelSpec:
    """자동 라벨링에 사용할 YOLO 모델 1개의 설정."""

    name: str
    path: Path
    conf: float
    iou: float
    class_map: dict[int, int]


@dataclass(frozen=True)
class Detection:
    """YOLO 탐지 결과 1개를 최종 class id 기준으로 표현한다."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float


def load_config(config_path: Path) -> dict[str, Any]:
    """YAML 파일에서 자동 라벨링 설정을 읽고 필수 키를 검증한다."""
    if not config_path.exists():
        raise FileNotFoundError(f"config_path does not exist: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")

    required_keys = [
        "models",
        "source_dir",
        "labels_auto_dir",
        "previews_dir",
        "duplicate_iou",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"config is missing required keys: {missing_keys}")

    return config


def resolve_path(base_dir: Path, path_value: str | Path) -> Path:
    """설정 파일 기준 상대 경로를 실제 Path로 변환한다."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def normalize_class_map(raw_class_map: Any) -> dict[int, int]:
    """설정 파일의 class_map을 int -> int 매핑으로 정규화한다."""
    if raw_class_map is None:
        return {}
    if not isinstance(raw_class_map, dict):
        raise ValueError("class_map must be a YAML mapping")

    class_map: dict[int, int] = {}
    for source_class_id, target_class_id in raw_class_map.items():
        class_map[int(source_class_id)] = int(target_class_id)
    return class_map


def normalize_threshold(raw_value: Any, field_name: str) -> float:
    """설정 파일의 임계값을 0.0-1.0 범위 float로 정규화한다."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return value


def build_model_specs(config: dict[str, Any], base_dir: Path) -> list[ModelSpec]:
    """설정값에서 실행할 YOLO 모델 목록을 만든다."""
    raw_models = config["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty YAML list")

    model_specs: list[ModelSpec] = []
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ValueError(f"models[{index}] must be a YAML mapping")
        if raw_model.get("enabled", True) is False:
            continue
        required_model_keys = ["path", "conf", "iou", "class_map"]
        missing_model_keys = [
            key for key in required_model_keys if key not in raw_model
        ]
        if missing_model_keys:
            raise ValueError(
                f"models[{index}] is missing required keys: {missing_model_keys}"
            )

        model_path = resolve_path(base_dir, raw_model["path"])
        conf = normalize_threshold(raw_model["conf"], f"models[{index}].conf")
        iou = normalize_threshold(raw_model["iou"], f"models[{index}].iou")
        class_map = normalize_class_map(raw_model.get("class_map"))
        if not class_map:
            raise ValueError(f"models[{index}].class_map must not be empty")

        model_specs.append(
            ModelSpec(
                name=str(raw_model.get("name", model_path.stem)),
                path=model_path,
                conf=conf,
                iou=iou,
                class_map=class_map,
            )
        )

    if not model_specs:
        raise ValueError("at least one model must be enabled")

    return model_specs


def load_model(model_path: Path) -> YOLO:
    """model_path에 있는 YOLO .pt 모델을 로드한다."""
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")
    return YOLO(str(model_path))


def find_missing_model_paths(model_specs: list[ModelSpec]) -> list[ModelSpec]:
    """사용 설정된 모델 중 .pt 파일이 실제로 없는 것을 미리 찾는다."""
    return [
        model_spec for model_spec in model_specs if not model_spec.path.exists()
    ]


def load_models(model_specs: list[ModelSpec]) -> list[tuple[ModelSpec, YOLO]]:
    """여러 YOLO 모델 설정을 실제 YOLO 객체 목록으로 로드한다."""
    return [(model_spec, load_model(model_spec.path)) for model_spec in model_specs]


def iter_image_paths(source_dir: Path) -> list[Path]:
    """source_dir 아래의 지원 이미지 파일을 정렬된 목록으로 반환한다."""
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir is not a directory: {source_dir}")

    image_paths = [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def predict_image(model: YOLO, image_path: Path, conf: float, iou: float):
    """image_path 이미지 1장에 대해 YOLO 추론을 실행하고 results를 반환한다."""
    results = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
    return results


def convert_boxes_to_detections(
    boxes,
    class_map: dict[int, int],
) -> list[Detection]:
    """Ultralytics Boxes 객체를 최종 class id 기준 Detection 목록으로 변환한다."""
    if boxes is None or len(boxes) == 0:
        return []

    cls_values = boxes.cls.cpu().tolist()
    conf_values = boxes.conf.cpu().tolist()
    xywhn_values = boxes.xywhn.cpu().tolist()

    detections: list[Detection] = []
    for class_value, confidence, xywhn_value in zip(
        cls_values,
        conf_values,
        xywhn_values,
    ):
        source_class_id = int(class_value)
        if source_class_id not in class_map:
            continue
        class_id = class_map[source_class_id]
        x_center, y_center, width, height = xywhn_value
        detections.append(
            Detection(
                class_id=class_id,
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
                confidence=float(confidence),
            )
        )

    return detections


def calculate_iou(first: Detection, second: Detection) -> float:
    """정규화된 YOLO bbox 2개의 IoU를 계산한다."""
    first_left = first.x_center - first.width / 2
    first_top = first.y_center - first.height / 2
    first_right = first.x_center + first.width / 2
    first_bottom = first.y_center + first.height / 2

    second_left = second.x_center - second.width / 2
    second_top = second.y_center - second.height / 2
    second_right = second.x_center + second.width / 2
    second_bottom = second.y_center + second.height / 2

    intersection_left = max(first_left, second_left)
    intersection_top = max(first_top, second_top)
    intersection_right = min(first_right, second_right)
    intersection_bottom = min(first_bottom, second_bottom)

    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first.width) * max(0.0, first.height)
    second_area = max(0.0, second.width) * max(0.0, second.height)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def remove_duplicate_detections(
    detections: list[Detection],
    duplicate_iou: float,
) -> list[Detection]:
    """같은 class끼리 많이 겹치는 bbox는 confidence가 높은 것만 남긴다."""
    if not detections:
        return []

    kept_detections: list[Detection] = []
    sorted_detections = sorted(
        detections,
        key=lambda detection: detection.confidence,
        reverse=True,
    )

    for candidate in sorted_detections:
        is_duplicate = any(
            candidate.class_id == kept_detection.class_id
            and calculate_iou(candidate, kept_detection) >= duplicate_iou
            for kept_detection in kept_detections
        )
        if not is_duplicate:
            kept_detections.append(candidate)

    return sorted(
        kept_detections,
        key=lambda detection: (
            detection.class_id,
            detection.x_center,
            detection.y_center,
            detection.width,
            detection.height,
        ),
    )


def convert_detections_to_yolo_lines(detections: list[Detection]) -> list[str]:
    """Detection 목록을 YOLO txt 라인 목록으로 변환한다."""
    label_lines: list[str] = []
    for detection in detections:
        line = (
            f"{detection.class_id} "
            f"{detection.x_center:.6f} "
            f"{detection.y_center:.6f} "
            f"{detection.width:.6f} "
            f"{detection.height:.6f}"
        )
        label_lines.append(line)

    return label_lines


def write_label_file(output_label_path: Path, label_lines: list[str]) -> None:
    """label_lines를 output_label_path에 YOLO txt 파일로 저장한다."""
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    label_text = "\n".join(label_lines)
    if label_text:
        label_text += "\n"
    output_label_path.write_text(label_text, encoding="utf-8")


def generate_label_lines_for_image(
    image_path: Path,
    models: list[tuple[ModelSpec, YOLO]],
    duplicate_iou: float,
) -> tuple[list[str], dict[str, int]]:
    """이미지 1장을 모든 모델로 추론하고 중복 제거된 YOLO txt 라인을 만든다."""
    detections: list[Detection] = []
    for model_spec, model in models:
        results = predict_image(model, image_path, model_spec.conf, model_spec.iou)
        if not results:
            continue
        detections.extend(
            convert_boxes_to_detections(results[0].boxes, model_spec.class_map)
        )

    filtered_detections = remove_duplicate_detections(detections, duplicate_iou)
    label_lines = convert_detections_to_yolo_lines(filtered_detections)
    statistics = {
        "boxes_raw": len(detections),
        "boxes_removed_duplicates": len(detections) - len(filtered_detections),
        "boxes_total": len(label_lines),
    }
    return label_lines, statistics


def generate_auto_label_file(
    image_path: Path,
    output_label_path: Path,
    models: list[tuple[ModelSpec, YOLO]],
    duplicate_iou: float,
) -> dict[str, int]:
    """이미지 1장의 자동 bbox를 생성하고 YOLO txt 파일로 저장한다."""
    if not image_path.exists():
        raise FileNotFoundError(f"image_path does not exist: {image_path}")

    label_lines, statistics = generate_label_lines_for_image(
        image_path,
        models,
        duplicate_iou,
    )
    write_label_file(output_label_path, label_lines)
    return statistics


def draw_preview(image_path: Path, label_lines: list[str], classes: list[str], preview_path: Path) -> None:
    """YOLO txt 라인 기준으로 bbox가 그려진 preview 이미지를 저장한다."""
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to save preview images") from exc

    preview_image = read_image(image_path)
    image_height, image_width = preview_image.shape[:2]
    for label_line in label_lines:
        parts = label_line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO label line: {label_line}")

        class_id = int(parts[0])
        x_center, y_center, width, height = [float(value) for value in parts[1:]]
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        label = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
        cv2.rectangle(preview_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview_image,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    write_image(preview_path, preview_image)


def describe_image_failure(image_path: Path, exc: Exception) -> str:
    """추론 실패 원인을 사용자가 이해할 수 있는 문장으로 바꾼다.

    사진을 읽지 못한 경우 YOLO 내부 오류가 그대로 나와서 원인을 알기 어렵다.
    """
    try:
        read_image(image_path)
    except Exception as read_exc:
        return f"사진을 읽을 수 없습니다({read_exc})"
    return str(exc)


def run_autolabel(
    config_path: Path,
    skip_existing: bool = False,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> dict[str, Any]:
    """config_path 설정에 따라 모든 이미지의 자동 라벨링을 실행한다.

    이미지 1장이 실패해도 전체를 중단하지 않고 실패 목록에 모아 끝까지 진행한다.
    """
    config = load_config(config_path)
    base_dir = config_path.parent

    model_specs = build_model_specs(config, base_dir)
    missing_models = find_missing_model_paths(model_specs)
    if missing_models:
        missing_text = ", ".join(
            f"{model_spec.name}={model_spec.path}" for model_spec in missing_models
        )
        raise FileNotFoundError(f"model_path does not exist: {missing_text}")

    source_dir = resolve_path(base_dir, config["source_dir"])
    labels_auto_dir = resolve_path(base_dir, config["labels_auto_dir"])
    previews_dir = resolve_path(base_dir, config["previews_dir"])
    duplicate_iou = normalize_threshold(config["duplicate_iou"], "duplicate_iou")
    save_previews = bool(config.get("save_previews", True))
    classes = [str(class_name) for class_name in config.get("classes", [])]

    models = load_models(model_specs)
    image_paths = iter_image_paths(source_dir)

    summary: dict[str, Any] = {
        "models_total": len(models),
        "images_total": len(image_paths),
        "images_skipped": 0,
        "labels_written": 0,
        "boxes_raw_total": 0,
        "boxes_removed_duplicates": 0,
        "boxes_total": 0,
        "previews_written": 0,
        "failures": [],
    }

    for image_index, image_path in enumerate(image_paths, start=1):
        if progress_callback is not None:
            progress_callback(image_index, len(image_paths), image_path)

        relative_image_path = image_path.relative_to(source_dir)
        output_label_path = labels_auto_dir / relative_image_path.with_suffix(".txt")
        preview_path = previews_dir / relative_image_path.with_suffix(".jpg")

        if skip_existing and output_label_path.exists():
            summary["images_skipped"] += 1
            continue

        try:
            label_lines, image_statistics = generate_label_lines_for_image(
                image_path,
                models,
                duplicate_iou,
            )
            write_label_file(output_label_path, label_lines)
        except Exception as exc:
            summary["failures"].append(
                {"image": str(image_path), "error": describe_image_failure(image_path, exc)}
            )
            continue

        summary["labels_written"] += 1
        summary["boxes_raw_total"] += image_statistics["boxes_raw"]
        summary["boxes_removed_duplicates"] += image_statistics[
            "boxes_removed_duplicates"
        ]
        summary["boxes_total"] += image_statistics["boxes_total"]

        if save_previews:
            try:
                draw_preview(image_path, label_lines, classes, preview_path)
            except Exception as exc:
                summary["failures"].append(
                    {"image": str(preview_path), "error": f"preview: {exc}"}
                )
            else:
                summary["previews_written"] += 1

    return summary


def parse_args() -> argparse.Namespace:
    """명령줄 인수를 파싱한다."""
    parser = argparse.ArgumentParser(description="YOLO .pt 모델로 YOLO txt 라벨을 생성합니다.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("autolabel_config.yaml"),
        help="autolabel_config.yaml 경로",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="이미 txt가 있는 이미지는 건너뛴다(중단된 작업 이어서 실행).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="이미지별 진행 상황을 출력하지 않는다.",
    )
    return parser.parse_args()


def print_progress(image_index: int, images_total: int, image_path: Path) -> None:
    """처리 중인 이미지 번호와 이름을 한 줄로 갱신 출력한다."""
    percent = image_index * 100 // max(1, images_total)
    print(
        f"\r[{image_index}/{images_total}] {percent:3d}%  {image_path.name[:50]:<50}",
        end="",
        flush=True,
    )


def main() -> None:
    """CLI 실행 진입점."""
    args = parse_args()
    config_path = args.config.resolve()

    try:
        summary = run_autolabel(
            config_path,
            skip_existing=args.skip_existing,
            progress_callback=None if args.quiet else print_progress,
        )
    except Exception as exc:
        print(f"\n자동 라벨링을 시작하지 못했습니다: {exc}")
        raise SystemExit(1) from exc

    if not args.quiet:
        print()

    print("Auto-labeling complete")
    print(f"models_total={summary['models_total']}")
    print(f"images_total={summary['images_total']}")
    print(f"images_skipped={summary['images_skipped']}")
    print(f"labels_written={summary['labels_written']}")
    print(f"boxes_raw_total={summary['boxes_raw_total']}")
    print(f"boxes_removed_duplicates={summary['boxes_removed_duplicates']}")
    print(f"boxes_total={summary['boxes_total']}")
    print(f"previews_written={summary['previews_written']}")
    print(f"failures={len(summary['failures'])}")

    for failure in summary["failures"]:
        print(f"  실패: {failure['image']} -> {failure['error']}")


if __name__ == "__main__":
    main()
