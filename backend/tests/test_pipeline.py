from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sku_audit.models import RecognitionStatus, ZoneType
from sku_audit.pipeline import run_image_folder_audit


class ImageFolderAuditTests(unittest.TestCase):
    def test_groups_exact_and_uncertain_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exact = root / "sku_exact_areas"
            uncertain = root / "sku_uncertain_areas"
            exact.mkdir()
            uncertain.mkdir()
            (exact / "sku_exact_zone_1.jpg").write_bytes(b"fake")
            (uncertain / "sku_uncertain_zone_1.jpg").write_bytes(b"fake")

            report = run_image_folder_audit(root)

            self.assertEqual(report.summary.total_evidence_files, 2)
            self.assertEqual(report.summary.exact_zone_files, 1)
            self.assertEqual(report.summary.uncertain_zone_files, 1)
            self.assertEqual(report.observations[0].zone_type, ZoneType.EXACT)
            self.assertEqual(report.observations[1].status, RecognitionStatus.NEEDS_RETAKE)

    def test_groups_top_level_folders_as_trading_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / "ТТ Example"
            store.mkdir()
            (store / "IMG_0001.HEIC").write_bytes(b"fake")

            report = run_image_folder_audit(root)

            self.assertEqual(report.summary.store_count, 1)
            self.assertEqual(report.summary.market_photo_files, 1)
            self.assertEqual(report.summary.pending_analysis_observations, 1)
            self.assertEqual(report.store_summaries[0].store_name, "ТТ Example")
            self.assertEqual(report.observations[0].status, RecognitionStatus.PENDING_ANALYSIS)


if __name__ == "__main__":
    unittest.main()
