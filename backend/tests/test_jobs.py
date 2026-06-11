from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sku_audit.jobs import JobStore
from sku_audit.models import MediaKind


class JobStoreTests(unittest.TestCase):
    def test_creates_job_with_image_quality_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = root / "ТТ Test"
            media_dir.mkdir()
            image_path = media_dir / "photo.jpg"
            Image.new("RGB", (1200, 900), color=(120, 120, 120)).save(image_path)

            store = JobStore(root / "var")
            job = store.create_from_local_folder(media_dir, store_name="ТТ Test")

            self.assertEqual(job.status, "completed")
            self.assertEqual(job.summary["total_files"], 1)
            self.assertEqual(job.summary["image_files"], 1)
            self.assertEqual(job.media[0].media_kind, MediaKind.IMAGE)
            self.assertEqual(job.media[0].store_name, "ТТ Test")

            loaded = store.get(job.job_id)
            self.assertEqual(loaded.job_id, job.job_id)
            self.assertIn("SKU Processing Job", loaded.report_markdown)


if __name__ == "__main__":
    unittest.main()
