from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import sku_audit.app as app_module
from sku_audit.jobs import JobStore


class ApiTests(unittest.TestCase):
    def test_upload_job_endpoint_accepts_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "photo.jpg"
            Image.new("RGB", (1200, 900), color=(120, 120, 120)).save(image_path)

            original_store = app_module.job_store
            app_module.job_store = JobStore(tmp_path / "var")
            try:
                client = TestClient(app_module.app)
                with image_path.open("rb") as file:
                    response = client.post(
                        "/jobs/upload",
                        data={"store_name": "ТТ API"},
                        files={"files": ("photo.jpg", file, "image/jpeg")},
                    )
            finally:
                app_module.job_store = original_store

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["store_name"], "ТТ API")
            self.assertEqual(payload["summary"]["total_files"], 1)
            self.assertEqual(payload["summary"]["image_files"], 1)


if __name__ == "__main__":
    unittest.main()
