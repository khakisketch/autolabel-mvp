from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from review_status import (
    load_completed_image_keys,
    load_status_by_source,
    save_completed_image_keys,
    save_status_by_source,
    source_key_for,
)


class ReviewStatusTest(unittest.TestCase):
    def test_keeps_completed_lists_separate_per_source_dir(self) -> None:
        """이미지 폴더를 바꿔도 이전 폴더의 완료 표시가 섞이지 않아야 한다."""
        with TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "review_status.json"

            save_completed_image_keys(status_path, {"a.jpg"}, source_key="D:/현장1")
            save_completed_image_keys(status_path, {"b.jpg", "c.jpg"}, source_key="D:/현장2")

            self.assertEqual(
                load_completed_image_keys(status_path, "D:/현장1"), {"a.jpg"}
            )
            self.assertEqual(
                load_completed_image_keys(status_path, "D:/현장2"), {"b.jpg", "c.jpg"}
            )
            self.assertEqual(load_completed_image_keys(status_path, "D:/새폴더"), set())

    def test_reads_old_version_1_status_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "review_status.json"
            status_path.write_text(
                json.dumps({"version": 1, "completed_images": ["old.jpg"]}),
                encoding="utf-8",
            )

            self.assertEqual(load_completed_image_keys(status_path), {"old.jpg"})

    def test_broken_status_file_is_backed_up_instead_of_crashing(self) -> None:
        """상태 파일이 깨져도 앱이 죽지 않고 경고만 돌려줘야 한다."""
        with TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "review_status.json"
            status_path.write_text("{ 깨진 JSON", encoding="utf-8")

            status_by_source, warning = load_status_by_source(status_path)

            self.assertEqual(status_by_source, {})
            self.assertIsNotNone(warning)
            self.assertTrue(status_path.with_suffix(".json.broken").exists())

    def test_unsupported_version_is_recovered_as_warning(self) -> None:
        with TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "review_status.json"
            status_path.write_text(json.dumps({"version": 99}), encoding="utf-8")

            status_by_source, warning = load_status_by_source(status_path)

            self.assertEqual(status_by_source, {})
            self.assertIn("99", str(warning))

    def test_empty_source_entries_are_not_written(self) -> None:
        with TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "review_status.json"

            save_status_by_source(status_path, {"D:/빈폴더": set(), "D:/현장": {"a.jpg"}})

            status_data = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status_data["version"], 2)
            self.assertEqual(list(status_data["sources"]), ["D:/현장"])

    def test_source_key_is_stable_for_same_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "사진"
            source_dir.mkdir()

            self.assertEqual(
                source_key_for(source_dir),
                source_key_for(source_dir / "." ),
            )


if __name__ == "__main__":
    unittest.main()
