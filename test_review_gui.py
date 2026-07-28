from __future__ import annotations

import unittest

from review_gui import CLASS_BBOX_COLORS, class_bbox_color


class ReviewGuiTest(unittest.TestCase):
    def test_different_classes_use_different_bbox_colors(self) -> None:
        self.assertNotEqual(class_bbox_color(0), class_bbox_color(1))
        self.assertNotEqual(class_bbox_color(1), class_bbox_color(3))

    def test_same_class_always_uses_same_bbox_color(self) -> None:
        self.assertEqual(class_bbox_color(3), class_bbox_color(3))

    def test_color_palette_repeats_safely_for_large_class_id(self) -> None:
        self.assertEqual(
            class_bbox_color(len(CLASS_BBOX_COLORS)),
            class_bbox_color(0),
        )


if __name__ == "__main__":
    unittest.main()
