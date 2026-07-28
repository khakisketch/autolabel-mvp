from __future__ import annotations

import unittest

import torch

from auto_label import convert_boxes_to_detections


class FakeBoxes:
    def __init__(self) -> None:
        self.cls = torch.tensor([0.0, 3.0, 8.0])
        self.conf = torch.tensor([0.9, 0.8, 0.7])
        self.xywhn = torch.tensor(
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


if __name__ == "__main__":
    unittest.main()
