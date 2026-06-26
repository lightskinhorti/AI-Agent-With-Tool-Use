from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

router = APIRouter()

REPORTS_DIR = Path("data/reports")


@router.get("/list")
async def list_reports():
    if not REPORTS_DIR.exists():
        return {"reports": []}
    reports = []
    for f in sorted(REPORTS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        reports.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })
    return {"reports": reports}


@router.get("/view/{filename}", response_class=PlainTextResponse)
async def view_report(filename: str):
    safe = Path(filename).name
    path = REPORTS_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Report not found")
    return path.read_text(encoding="utf-8")


@router.get("/download/{filename}")
async def download_report(filename: str):
    safe = Path(filename).name
    path = REPORTS_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Report not found")
    return FileResponse(
        path=path, filename=safe, media_type="text/markdown"
    )
