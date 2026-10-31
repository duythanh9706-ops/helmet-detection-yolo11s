"""Unit test không cần file .pt, GPU, webcam hay thư viện cv2."""
import tempfile
from pathlib import Path
import unittest

from src.common import (CSV_FIELDS, DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU,
                        RunLogger, RunStats, category_from_name, get_images)


class TestCommon(unittest.TestCase):
    def test_defaults_follow_original_files(self):
        self.assertEqual(DEFAULT_CONF, 0.32)
        self.assertEqual(DEFAULT_IOU, 0.60)
        self.assertEqual(DEFAULT_IMGSZ, 640)

    def test_class_name_normalization(self):
        self.assertEqual(category_from_name("Helmet"), "helmet")
        self.assertEqual(category_from_name("no_helmet"), "no_helmet")
        self.assertEqual(category_from_name("No-Helmet"), "no_helmet")
        self.assertEqual(category_from_name("without helmet"), "no_helmet")
        self.assertEqual(category_from_name("unknown"), "other")

    def test_image_input(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "b.png").touch()
            (folder / "a.jpg").touch()
            (folder / "notes.txt").touch()
            self.assertEqual([p.name for p in get_images(folder)], ["a.jpg", "b.png"])

    def test_csv_logger(self):
        with tempfile.TemporaryDirectory() as temp:
            log = RunLogger(temp, "image")
            self.assertEqual(len(CSV_FIELDS), 10)
            log.write("test.jpg", 1, [{"class_name": "helmet", "confidence": 0.95,
                                       "x1": 1, "y1": 2, "x2": 3, "y2": 4}])
            stats = RunStats("image", "test.jpg")
            stats.add({"helmet": 1, "no_helmet": 0, "other": 0})
            log.close(stats)
            self.assertTrue(log.csv_path.exists())
            self.assertIn("helmet", log.csv_path.read_text(encoding="utf-8-sig"))
            self.assertTrue(log.summary_path.exists())


if __name__ == "__main__":
    unittest.main()
