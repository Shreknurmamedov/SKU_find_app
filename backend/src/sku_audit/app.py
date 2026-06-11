from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from sku_audit.pipeline import run_image_folder_audit

app = FastAPI(title="SKU Audit Backend", version="0.1.0")


class ImageAuditRequest(BaseModel):
    input_root: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/audit/images")
def audit_images(request: ImageAuditRequest) -> dict:
    report = run_image_folder_audit(Path(request.input_root))
    return report.to_dict()
