from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from review_labels import (
    BBox,
    BBoxHistory,
    bbox_from_canvas_rect,
    bbox_to_canvas_rect,
    change_bbox_class,
    create_bbox,
    delete_bbox,
    load_bboxes,
    resolve_label_for_review,
    save_bboxes,
    update_bbox,
)
from review_status import (
    find_next_uncompleted_index,
    load_completed_image_keys,
    save_completed_image_keys,
)


class ReviewLabelsTest(unittest.TestCase):
    def test_save_and_load_bboxes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "labels_reviewed" / "sample.txt"
            bboxes = [
                BBox(0, 0.500000, 0.500000, 0.200000, 0.300000),
                BBox(1, 0.250000, 0.250000, 0.100000, 0.100000),
            ]

            save_bboxes(label_path, bboxes)

            self.assertEqual(load_bboxes(label_path), bboxes)
            self.assertEqual(
                label_path.read_text(encoding="utf-8"),
                (
                    "0 0.500000 0.500000 0.200000 0.300000\n"
                    "1 0.250000 0.250000 0.100000 0.100000\n"
                ),
            )

    def test_create_update_delete_bbox(self) -> None:
        first = BBox(0, 0.5, 0.5, 0.2, 0.2)
        second = BBox(1, 0.3, 0.3, 0.1, 0.1)

        bboxes = create_bbox([], first)
        self.assertEqual(bboxes, [first])

        bboxes = update_bbox(bboxes, 0, second)
        self.assertEqual(bboxes, [second])

        bboxes = delete_bbox(bboxes, 0)
        self.assertEqual(bboxes, [])

    def test_bbox_history_undo_and_redo(self) -> None:
        history = BBoxHistory()
        original_bboxes = [BBox(0, 0.5, 0.5, 0.2, 0.2)]
        changed_bboxes = [BBox(0, 0.6, 0.5, 0.2, 0.2)]

        history.record(original_bboxes)

        undone_bboxes = history.undo(changed_bboxes)
        redone_bboxes = history.redo(undone_bboxes or [])

        self.assertEqual(undone_bboxes, original_bboxes)
        self.assertEqual(redone_bboxes, changed_bboxes)

    def test_bbox_history_new_change_clears_redo(self) -> None:
        history = BBoxHistory()
        first_bboxes = [BBox(0, 0.5, 0.5, 0.2, 0.2)]
        second_bboxes = [BBox(0, 0.6, 0.5, 0.2, 0.2)]

        history.record(first_bboxes)
        history.undo(second_bboxes)
        history.record(first_bboxes)

        self.assertIsNone(history.redo(first_bboxes))

    def test_change_bbox_class(self) -> None:
        bboxes = [BBox(0, 0.5, 0.5, 0.2, 0.2)]

        changed_bboxes = change_bbox_class(bboxes, 0, 1)

        self.assertEqual(changed_bboxes, [BBox(1, 0.5, 0.5, 0.2, 0.2)])

    def test_canvas_rect_conversion(self) -> None:
        bbox = bbox_from_canvas_rect(
            class_id=1,
            left=100,
            top=50,
            right=300,
            bottom=250,
            image_width=1000,
            image_height=500,
        )

        self.assertEqual(bbox, BBox(1, 0.2, 0.3, 0.2, 0.4))
        rect = bbox_to_canvas_rect(bbox, 1000, 500)
        expected_rect = (100.0, 50.0, 300.0, 250.0)
        for actual_value, expected_value in zip(rect, expected_rect):
            self.assertAlmostEqual(actual_value, expected_value)

    def test_reviewed_label_has_priority(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reviewed_label_path = root / "labels_reviewed" / "sample.txt"
            auto_label_path = root / "labels_auto" / "sample.txt"
            reviewed_label_path.parent.mkdir(parents=True)
            reviewed_label_path.write_text(
                "0 0.500000 0.500000 0.200000 0.200000\n",
                encoding="utf-8",
            )
            create_calls = []

            selected_path = resolve_label_for_review(
                reviewed_label_path,
                auto_label_path,
                lambda: create_calls.append(True),
            )

            self.assertEqual(selected_path, reviewed_label_path)
            self.assertEqual(create_calls, [])

    def test_missing_label_creates_auto_label(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reviewed_label_path = root / "labels_reviewed" / "sample.txt"
            auto_label_path = root / "labels_auto" / "sample.txt"

            def create_auto_label() -> None:
                auto_label_path.parent.mkdir(parents=True)
                auto_label_path.write_text(
                    "1 0.250000 0.250000 0.100000 0.100000\n",
                    encoding="utf-8",
                )

            selected_path = resolve_label_for_review(
                reviewed_label_path,
                auto_label_path,
                create_auto_label,
            )

            self.assertEqual(selected_path, auto_label_path)
            self.assertTrue(auto_label_path.exists())

    def test_save_and_load_completed_image_keys(self) -> None:
        with TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "labels_reviewed" / "review_status.json"
            completed_image_keys = {"flight_02/b.jpg", "flight_01/a.jpg"}

            save_completed_image_keys(status_path, completed_image_keys)

            self.assertEqual(
                load_completed_image_keys(status_path),
                completed_image_keys,
            )
            self.assertIn(
                '"completed_images": [',
                status_path.read_text(encoding="utf-8"),
            )

    def test_find_next_uncompleted_index_wraps_around(self) -> None:
        image_keys = ["a.jpg", "b.jpg", "c.jpg"]

        next_index = find_next_uncompleted_index(
            image_keys,
            {"b.jpg", "c.jpg"},
            current_index=1,
        )

        self.assertEqual(next_index, 0)

    def test_find_next_uncompleted_index_returns_none_when_all_complete(self) -> None:
        image_keys = ["a.jpg", "b.jpg"]

        next_index = find_next_uncompleted_index(
            image_keys,
            {"a.jpg", "b.jpg"},
            current_index=0,
        )

        self.assertIsNone(next_index)


if __name__ == "__main__":
    unittest.main()
