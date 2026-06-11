from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from sku_audit.catalog import load_catalog
from sku_audit.jobs import JobStore
from sku_audit.pipeline import run_image_folder_audit

app = FastAPI(title="SKU Audit Backend", version="0.1.0")
job_store = JobStore(Path(os.environ.get("SKU_AUDIT_VAR_DIR", "var")))


class ImageAuditRequest(BaseModel):
    input_root: str


class LocalJobRequest(BaseModel):
    input_root: str
    store_name: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/audit/images")
def audit_images(request: ImageAuditRequest) -> dict:
    report = run_image_folder_audit(Path(request.input_root))
    return report.to_dict()


@app.get("/catalog/summary")
def catalog_summary() -> dict:
    catalog_path = Path("..") / "data" / "catalog" / "own_products.csv"
    if not catalog_path.exists():
        catalog_path = Path("data") / "catalog" / "own_products.csv"
    entries = load_catalog(catalog_path)
    brands: dict[str, int] = {}
    for entry in entries:
        brands[entry.brand_name] = brands.get(entry.brand_name, 0) + 1
    return {
        "catalog_path": str(catalog_path),
        "total_skus": len(entries),
        "brands": dict(sorted(brands.items(), key=lambda item: item[1], reverse=True)),
    }


@app.post("/jobs/from-local")
def create_job_from_local(request: LocalJobRequest) -> dict:
    input_root = Path(request.input_root)
    if not input_root.exists():
        raise HTTPException(status_code=404, detail="Input folder does not exist")
    job = job_store.create_from_local_folder(input_root, store_name=request.store_name)
    return job.to_dict()


@app.post("/jobs/upload")
async def upload_job(
    store_name: str | None = Form(default=None),
    files: list[UploadFile] = File(...),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    payloads: list[tuple[str, bytes]] = []
    for file in files:
        content = await file.read()
        if not content:
            continue
        payloads.append((file.filename or "upload.bin", content))
    if not payloads:
        raise HTTPException(status_code=400, detail="All uploaded files were empty")
    job = job_store.save_uploaded_bytes(payloads, store_name=store_name)
    return job.to_dict()


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return job_store.list_jobs()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return job_store.get(job_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.get("/jobs/{job_id}/report.md")
def get_job_report(job_id: str) -> Response:
    try:
        job = job_store.get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return Response(content=job.report_markdown, media_type="text/markdown; charset=utf-8")
