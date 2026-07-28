from __future__ import annotations

import json
from pathlib import Path


STATUS_FILE_VERSION = 1


def load_completed_image_keys(status_path: Path) -> set[str]:
    """검수 상태 JSON에서 검수 완료된 이미지 상대 경로를 읽는다."""
    if not status_path.exists():
        return set()

    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(status_data, dict):
        raise ValueError("review status must be a JSON object")
    if status_data.get("version") != STATUS_FILE_VERSION:
        raise ValueError("unsupported review status version")

    completed_images = status_data.get("completed_images", [])
    if not isinstance(completed_images, list) or not all(
        isinstance(image_key, str) for image_key in completed_images
    ):
        raise ValueError("completed_images must be a JSON string list")
    return set(completed_images)


def save_completed_image_keys(status_path: Path, completed_image_keys: set[str]) -> None:
    """검수 완료 이미지 상대 경로를 정렬된 JSON 파일로 저장한다."""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_data = {
        "version": STATUS_FILE_VERSION,
        "completed_images": sorted(completed_image_keys),
    }
    status_path.write_text(
        json.dumps(status_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def find_next_uncompleted_index(
    image_keys: list[str],
    completed_image_keys: set[str],
    current_index: int,
) -> int | None:
    """현재 위치 다음부터 순환하며 검수 완료되지 않은 이미지 번호를 찾는다."""
    if not image_keys:
        return None
    if current_index < 0 or current_index >= len(image_keys):
        raise IndexError(f"current image index out of range: {current_index}")

    for offset in range(1, len(image_keys) + 1):
        candidate_index = (current_index + offset) % len(image_keys)
        if image_keys[candidate_index] not in completed_image_keys:
            return candidate_index
    return None
