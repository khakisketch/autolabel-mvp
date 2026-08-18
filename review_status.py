from __future__ import annotations

import json
from pathlib import Path


STATUS_FILE_VERSION = 2
LEGACY_STATUS_FILE_VERSION = 1
DEFAULT_SOURCE_KEY = ""


def source_key_for(source_dir: Path) -> str:
    """이미지 폴더 경로를 검수 상태 파일 안에서 쓸 키로 바꾼다."""
    return source_dir.resolve().as_posix()


def load_status_by_source(status_path: Path) -> tuple[dict[str, set[str]], str | None]:
    """폴더별 검수 완료 목록을 읽는다.

    파일이 깨져 있어도 예외로 앱을 죽이지 않고, 깨진 파일을 백업한 뒤
    빈 상태와 사용자에게 보여줄 경고 문구를 함께 돌려준다.
    """
    if not status_path.exists():
        return {}, None

    try:
        status_data = json.loads(status_path.read_text(encoding="utf-8"))
        return parse_status_data(status_data), None
    except Exception as exc:
        backup_path = backup_broken_status(status_path)
        warning = (
            f"검수 상태 파일을 읽지 못해 새로 시작합니다: {exc}\n"
            f"기존 파일은 {backup_path.name}(으)로 옮겼습니다."
        )
        return {}, warning


def parse_status_data(status_data: object) -> dict[str, set[str]]:
    """검수 상태 JSON 구조를 폴더별 완료 목록으로 변환한다."""
    if not isinstance(status_data, dict):
        raise ValueError("review status must be a JSON object")

    version = status_data.get("version")
    if version == LEGACY_STATUS_FILE_VERSION:
        return {DEFAULT_SOURCE_KEY: parse_image_keys(status_data.get("completed_images", []))}
    if version != STATUS_FILE_VERSION:
        raise ValueError(f"unsupported review status version: {version}")

    raw_sources = status_data.get("sources", {})
    if not isinstance(raw_sources, dict):
        raise ValueError("sources must be a JSON object")

    return {
        str(source_key): parse_image_keys(image_keys)
        for source_key, image_keys in raw_sources.items()
    }


def parse_image_keys(raw_image_keys: object) -> set[str]:
    """완료 이미지 목록이 문자열 배열인지 확인하고 집합으로 만든다."""
    if not isinstance(raw_image_keys, list) or not all(
        isinstance(image_key, str) for image_key in raw_image_keys
    ):
        raise ValueError("completed_images must be a JSON string list")
    return set(raw_image_keys)


def backup_broken_status(status_path: Path) -> Path:
    """읽을 수 없는 검수 상태 파일을 덮어쓰기 전에 백업해 둔다."""
    backup_path = status_path.with_suffix(status_path.suffix + ".broken")
    try:
        if backup_path.exists():
            backup_path.unlink()
        status_path.rename(backup_path)
    except OSError:
        return status_path
    return backup_path


def save_status_by_source(
    status_path: Path,
    status_by_source: dict[str, set[str]],
) -> None:
    """폴더별 검수 완료 목록을 정렬된 JSON 파일로 저장한다."""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_data = {
        "version": STATUS_FILE_VERSION,
        "sources": {
            source_key: sorted(image_keys)
            for source_key, image_keys in sorted(status_by_source.items())
            if image_keys
        },
    }
    status_path.write_text(
        json.dumps(status_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_completed_image_keys(
    status_path: Path,
    source_key: str = DEFAULT_SOURCE_KEY,
) -> set[str]:
    """검수 상태 파일에서 한 이미지 폴더의 완료 목록만 읽는다."""
    status_by_source, _warning = load_status_by_source(status_path)
    if source_key not in status_by_source and DEFAULT_SOURCE_KEY in status_by_source:
        return status_by_source[DEFAULT_SOURCE_KEY]
    return status_by_source.get(source_key, set())


def save_completed_image_keys(
    status_path: Path,
    completed_image_keys: set[str],
    source_key: str = DEFAULT_SOURCE_KEY,
) -> None:
    """한 이미지 폴더의 완료 목록만 갱신해 저장한다."""
    status_by_source, _warning = load_status_by_source(status_path)
    status_by_source[source_key] = set(completed_image_keys)
    save_status_by_source(status_path, status_by_source)


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
