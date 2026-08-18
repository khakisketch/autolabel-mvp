from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(image_path: Path):
    """한글·공백이 포함된 경로에서도 안전하게 이미지를 읽는다.

    cv2.imread는 Windows에서 ANSI 경로만 처리하므로 한글 폴더의 이미지를
    조용히 None으로 반환한다. 파일을 바이트로 직접 읽고 디코딩해서 이 문제를 피한다.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"image_path does not exist: {image_path}")

    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(f"failed to read image file: {image_path}") from exc

    if image_bytes.size == 0:
        raise RuntimeError(f"image file is empty: {image_path}")

    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode image: {image_path}")
    return image


def write_image(image_path: Path, image) -> None:
    """한글·공백이 포함된 경로에도 이미지를 저장하고 실패를 예외로 알린다.

    cv2.imwrite는 실패해도 예외 없이 False만 반환해서 저장 실패가 조용히 묻힌다.
    """
    image_path.parent.mkdir(parents=True, exist_ok=True)
    extension = image_path.suffix if image_path.suffix else ".jpg"

    success, encoded_image = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"failed to encode image: {image_path}")

    try:
        encoded_image.tofile(str(image_path))
    except OSError as exc:
        raise RuntimeError(f"failed to write image file: {image_path}") from exc
