from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from src.cockpit.daily import PROJECT_ROOT, build_cockpit_payload, build_recommendations, build_status

router = APIRouter(tags=["Daily Cockpit"])


@router.get("/cockpit", response_class=HTMLResponse, include_in_schema=False)
def cockpit_page() -> HTMLResponse:
    html_path = PROJECT_ROOT / "static" / "cockpit" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Daily cockpit asset not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/api/cockpit/summary")
def cockpit_summary(limit: int = Query(30, ge=1, le=200)) -> dict:
    return build_cockpit_payload(limit=limit)


@router.get("/api/cockpit/status")
def cockpit_status() -> dict:
    return build_status()


@router.get("/api/cockpit/recommendations")
def cockpit_recommendations(limit: int = Query(30, ge=1, le=200)) -> dict:
    return {"recommendations": build_recommendations(limit=limit)}

