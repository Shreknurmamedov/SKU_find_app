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
            original_start = app_module._start_sku_counting
            app_module.job_store = JobStore(tmp_path / "var")
            app_module._start_sku_counting = lambda job_id: None
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
                app_module._start_sku_counting = original_start

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["store_name"], "ТТ API")
            self.assertEqual(payload["summary"]["total_files"], 1)
            self.assertEqual(payload["summary"]["image_files"], 1)

    def test_upload_job_endpoint_saves_feedback_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "photo.jpg"
            feedback_path = tmp_path / "hardneg.jpg"
            sidecar_path = tmp_path / "hardneg.json"
            Image.new("RGB", (1200, 900), color=(120, 120, 120)).save(image_path)
            Image.new("RGB", (128, 128), color=(40, 180, 80)).save(feedback_path)
            sidecar_path.write_text('{"label":"hard_negative"}', encoding="utf-8")

            original_store = app_module.job_store
            original_start = app_module._start_sku_counting
            app_module.job_store = JobStore(tmp_path / "var")
            app_module._start_sku_counting = lambda job_id: None
            try:
                client = TestClient(app_module.app)
                with image_path.open("rb") as image_file, feedback_path.open("rb") as feedback_file, sidecar_path.open("rb") as sidecar_file:
                    response = client.post(
                        "/jobs/upload",
                        data={"store_name": "TT API"},
                        files=[
                            ("files", ("photo.jpg", image_file, "image/jpeg")),
                            ("feedback_files", ("hardneg.jpg", feedback_file, "image/jpeg")),
                            ("feedback_files", ("hardneg.json", sidecar_file, "application/json")),
                        ],
                    )
            finally:
                app_module.job_store = original_store
                app_module._start_sku_counting = original_start

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]
            feedback_dir = tmp_path / "var" / "uploads" / job_id / "feedback"
            self.assertTrue((feedback_dir / "hardneg.jpg").exists())
            self.assertTrue((feedback_dir / "hardneg.json").exists())


if __name__ == "__main__":
    unittest.main()
