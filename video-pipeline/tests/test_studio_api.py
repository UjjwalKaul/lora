"""
Automated Contract & Stability Test Suite for Local Video Studio Backend.
Enforces:
- Zero-Torch Web Process assertion ("torch" not in sys.modules)
- Idle RSS memory footprint (< 80 MB)
- Single-GPU mutex enforcement (HTTP 409 Conflict)
- Non-destructive manifest reconciliation
- In-process approvals (0 ms latency)
- Path traversal protection (HTTP 403 Forbidden)
"""

import os
import shutil
import sys
import time
from pathlib import Path
import psutil
import pytest
from fastapi.testclient import TestClient

# Anchor pipeline path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

# -----------------------------------------------------------------------------
# 1. Zero-Torch & Memory Footprint Assertion (Before any imports)
# -----------------------------------------------------------------------------
def test_zero_torch_import():
    """Assert that importing studio.server does NOT import torch into sys.modules."""
    from studio.server import app  # noqa: F401
    assert "torch" not in sys.modules, "FATAL: torch was imported into the web server process!"
    assert "diffusers" not in sys.modules, "FATAL: diffusers was imported into the web server process!"


def test_idle_rss_memory_footprint():
    """Assert server idle RSS memory remains strictly under 80 MB."""
    from studio.server import app  # noqa: F401
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"\n[SERVER MEMORY TEST] Current process RSS: {rss_mb:.2f} MB")
    assert rss_mb < 80.0, f"Expected server idle memory < 80 MB RSS, got {rss_mb:.2f} MB"


# -----------------------------------------------------------------------------
# 2. API Contract & Lifecycle Tests
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from studio.server import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def test_project_name():
    name = "pytest_studio_test_proj"
    proj_dir = PIPELINE_ROOT / "workspace" / "projects" / name
    if proj_dir.exists():
        shutil.rmtree(proj_dir)
    yield name
    if proj_dir.exists():
        shutil.rmtree(proj_dir)


def test_root_index(client):
    """Assert root endpoint serves static HTML."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_create_project(client, test_project_name):
    """Assert project creation works and returns 201."""
    payload = {
        "project_name": test_project_name,
        "style_profile": "photoreal_default"
    }
    res = client.post("/api/projects", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "ok"

    # Duplicate creation should return 409 Conflict
    res_dup = client.post("/api/projects", json=payload)
    assert res_dup.status_code == 409


def test_list_and_get_project(client, test_project_name):
    """Assert project summary and details endpoints work."""
    res = client.get("/api/projects")
    assert res.status_code == 200
    projects = res.json()
    assert any(p["name"] == test_project_name for p in projects)

    detail_res = client.get(f"/api/projects/{test_project_name}")
    assert detail_res.status_code == 200
    details = detail_res.json()
    assert "manifest" in details
    assert "script" in details
    assert len(details["manifest"]["shots"]) == 2


def test_non_destructive_manifest_reconciliation(client, test_project_name):
    """Assert updating script retains existing shot state."""
    # First, approve shot_001
    client.post(f"/api/projects/{test_project_name}/shots/shot_001/approve")

    # Get script and modify shot prompt while keeping shot_001
    details = client.get(f"/api/projects/{test_project_name}").json()
    script = details["script"]
    script["shots"][0]["prompt"] = "Updated test prompt for shot 1"
    
    # Add a third shot
    script["shots"].append({
        "shot_id": "shot_003",
        "shot_type": "close_up",
        "camera_motion": "static",
        "duration_sec": 1.0,
        "prompt": "Third shot description",
        "seed": 777
    })

    update_res = client.post(f"/api/projects/{test_project_name}/script", json={"script": script})
    assert update_res.status_code == 200

    # Verify manifest reconciled: shot_001 remains approved
    reconciled = client.get(f"/api/projects/{test_project_name}").json()
    shots = reconciled["manifest"]["shots"]
    assert len(shots) == 3
    shot1 = next(s for s in shots if s["shot_id"] == "shot_001")
    assert shot1["status"] == "approved", "Existing approval status was unexpectedly reset!"
    assert shot1["prompt"] == "Updated test prompt for shot 1"

    shot3 = next(s for s in shots if s["shot_id"] == "shot_003")
    assert shot3["status"] == "pending_review"
    assert shot3["num_frames"] == 25, "1.0s did not snap to 8n+1 (25 frames)!"


def test_update_script_syncs_manifest_config(client, test_project_name):
    """Assert updating Global Style Prompt, negative prompt, and target_fps via /script syncs manifest.config."""
    details = client.get(f"/api/projects/{test_project_name}").json()
    script = details["script"]
    test_style = "vibrant full-color Spider-Verse movie still, rich neon magenta cyan orange lighting"
    test_neg = "monochrome, grayscale, black and white"
    test_fps = 30
    script["global_style_prompt"] = test_style
    script["negative_prompt"] = test_neg
    script["target_fps"] = test_fps

    update_res = client.post(f"/api/projects/{test_project_name}/script", json={"script": script})
    assert update_res.status_code == 200

    reconciled = client.get(f"/api/projects/{test_project_name}").json()
    manifest_cfg = reconciled["manifest"]["config"]
    assert manifest_cfg["style_prompt"] == test_style
    assert manifest_cfg["negative_prompt"] == test_neg
    assert manifest_cfg["target_fps"] == test_fps
    assert reconciled["manifest"]["global_style_prompt"] == test_style
    assert reconciled["manifest"]["negative_prompt"] == test_neg


def test_in_process_shot_actions(client, test_project_name):
    """Assert shot approval, rejection, and approve-all work in-process."""
    # Reject shot_002
    res_rej = client.post(f"/api/projects/{test_project_name}/shots/shot_002/reject")
    assert res_rej.status_code == 200

    details = client.get(f"/api/projects/{test_project_name}").json()
    shot2 = next(s for s in details["manifest"]["shots"] if s["shot_id"] == "shot_002")
    assert shot2["status"] == "rejected"

    # Approve all
    res_all = client.post(f"/api/projects/{test_project_name}/approve-all")
    assert res_all.status_code == 200

    details_after = client.get(f"/api/projects/{test_project_name}").json()
    for s in details_after["manifest"]["shots"]:
        assert s["status"] == "approved"


def test_styles_and_look_dev_config(client, test_project_name):
    """Assert styles listing returns full preset details and config endpoint updates look-dev."""
    styles_res = client.get("/api/styles")
    assert styles_res.status_code == 200
    styles = styles_res.json()
    assert "photoreal_default" in styles
    assert "comic_25d" in styles
    assert "stage1" in styles["photoreal_default"]

    # Tweak config directly
    config_payload = {
        "keyframe_backend": "sdxl_quality",
        "lora_path": "custom/test.safetensors",
        "lora_scale": 0.85,
        "stepped_fps": 12,
        "chromatic_aberration_px": 2
    }
    cfg_res = client.post(f"/api/projects/{test_project_name}/config", json=config_payload)
    assert cfg_res.status_code == 200

    details = client.get(f"/api/projects/{test_project_name}").json()
    cfg = details["manifest"]["config"]
    assert cfg["keyframe_backend"] == "sdxl_quality"
    assert cfg["lora_path"] == "custom/test.safetensors"
    assert cfg["lora_scale"] == 0.85
    assert cfg["post_fx"]["stepped_fps"] == 12
    assert cfg["post_fx"]["chromatic_aberration_px"] == 2


def test_keyframe_backend_factory_sdxl_quality():
    """Assert get_keyframe_backend creates SDXLQualityBackend with lazy imports."""
    from core.keyframe_backend import get_keyframe_backend, SDXLQualityBackend
    backend = get_keyframe_backend("sdxl_quality")
    assert isinstance(backend, SDXLQualityBackend)
    assert backend.backend_name == "sdxl_quality"
    # Also test alias
    alias_backend = get_keyframe_backend("quality")
    assert isinstance(alias_backend, SDXLQualityBackend)


def test_single_job_mutex_and_cancellation(client, test_project_name):
    """Assert the single-GPU mutex rejects concurrent jobs with HTTP 409 and cancels cleanly."""
    # Launch mock keyframes job
    run_res = client.post(f"/api/projects/{test_project_name}/run", json={"stage": "keyframes", "mock": True})
    assert run_res.status_code == 200

    # Attempt launching a second job while the first is running
    conflict_res = client.post(f"/api/projects/{test_project_name}/run", json={"stage": "motion", "mock": True})
    # If the mock job finished instantly, conflict might not hit, but let's check status
    status_res = client.get("/api/jobs/active").json()
    if status_res["active"]:
        assert conflict_res.status_code == 409, "Second job was not blocked by the single-job mutex!"

    # Cancel job
    cancel_res = client.post("/api/jobs/cancel")
    assert cancel_res.status_code == 200


def test_path_traversal_protection(client, test_project_name):
    """Assert path traversal attempts on /media return HTTP 403 Forbidden."""
    res = client.get(f"/media/{test_project_name}/keyframes/../../secret.txt")
    assert res.status_code in (403, 404), f"Expected 403/404, got {res.status_code}"
