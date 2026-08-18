from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from image_io import read_image, write_image


class ImageIoTest(unittest.TestCase):
    """한글 경로에서 cv2.imread/imwrite가 실패하던 문제를 고정한다."""

    def test_reads_and_writes_image_under_korean_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            korean_dir = Path(temp_dir) / "공사현장 사진"
            image_path = korean_dir / "드론_01.png"
            source_image = np.zeros((8, 12, 3), dtype=np.uint8)
            source_image[2, 3] = (10, 20, 30)

            write_image(image_path, source_image)
            loaded_image = read_image(image_path)

            self.assertTrue(image_path.exists())
            self.assertEqual(loaded_image.shape, source_image.shape)
            self.assertEqual(tuple(loaded_image[2, 3]), (10, 20, 30))

    def test_missing_image_raises_file_not_found(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                read_image(Path(temp_dir) / "없는파일.jpg")

    def test_broken_image_raises_instead_of_returning_none(self) -> None:
        with TemporaryDirectory() as temp_dir:
            broken_path = Path(temp_dir) / "깨진사진.jpg"
            broken_path.write_bytes(b"this is not an image")

            with self.assertRaises(RuntimeError):
                read_image(broken_path)

    def test_empty_image_file_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / "빈파일.jpg"
            empty_path.write_bytes(b"")

            with self.assertRaises(RuntimeError):
                read_image(empty_path)


if __name__ == "__main__":
    unittest.main()
