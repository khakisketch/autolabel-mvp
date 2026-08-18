from __future__ import annotations

from pathlib import Path
import unittest

from auto_label import (
    Detection,
    ModelSpec,
    convert_boxes_to_detections,
    find_missing_model_paths,
    remove_duplicate_detections,
)


class FakeTensor:
    """torch 설치 없이도 Boxes 인터페이스를 흉내 내는 최소 구현."""

    def __init__(self, values: list) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list:
        return self.values


class FakeBoxes:
    def __init__(self) -> None:
        self.cls = FakeTensor([0.0, 3.0, 8.0])
        self.conf = FakeTensor([0.9, 0.8, 0.7])
        self.xywhn = FakeTensor(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.4, 0.4, 0.3, 0.3],
                [0.3, 0.3, 0.1, 0.1],
            ]
        )

    def __len__(self) -> int:
        return 3


class AutoLabelTest(unittest.TestCase):
    def test_converts_equipment_classes_to_final_class_ids(self) -> None:
        detections = convert_boxes_to_detections(
            FakeBoxes(),
            {
                0: 1,
                3: 4,
                8: 9,
            },
        )

        self.assertEqual(len(detections), 3)
        self.assertEqual(detections[0].class_id, 1)
        self.assertEqual(detections[1].class_id, 4)
        self.assertEqual(detections[2].class_id, 9)

    def test_ignores_classes_not_selected_in_class_map(self) -> None:
        detections = convert_boxes_to_detections(FakeBoxes(), {3: 4})

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 4)

    def test_keeps_highest_confidence_box_among_duplicates(self) -> None:
        detections = [
            Detection(0, 0.5, 0.5, 0.2, 0.2, 0.6),
            Detection(0, 0.5, 0.5, 0.2, 0.2, 0.95),
            Detection(1, 0.5, 0.5, 0.2, 0.2, 0.7),
        ]

        kept = remove_duplicate_detections(detections, duplicate_iou=0.9)

        self.assertEqual(len(kept), 2)
        self.assertEqual({detection.class_id for detection in kept}, {0, 1})
        self.assertEqual(
            [detection.confidence for detection in kept if detection.class_id == 0],
            [0.95],
        )

    def test_finds_missing_model_files_before_running(self) -> None:
        """모델 .pt가 없으면 첫 추론 때가 아니라 시작할 때 알려줘야 한다."""
        model_specs = [
            ModelSpec("person", Path("없는모델.pt"), 0.25, 0.7, {0: 0}),
        ]

        missing = find_missing_model_paths(model_specs)

        self.assertEqual([spec.name for spec in missing], ["person"])


if __name__ == "__main__":
    unittest.main()
