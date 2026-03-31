"""
main.py — floorplan-to-structured-2d-fbm (drywall_bq repo)
Migrated from BigQuery to PostgreSQL.
Keeps existing architecture (page_to_structured_2d, bounding_box_offsets, multi-client Vertex AI).
"""

import logging
import asyncio
import uuid
import time as time_module
from pathlib import Path
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from concurrent.futures import ThreadPoolExecutor

import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from google.cloud import secretmanager

from modeller_2d import FloorPlan2D
from helper import (
    enable_logging_on_stdout,
    create_pg_pool,
    pg_fetch_all,
    pg_fetch_one,
    pg_execute,
    log_json,
    timed_step,
    parse_jsonb,
    get_gcs_client,
    load_vertex_ai_client,
    load_gcp_credentials,
    load_hyperparameters,
    transcribe,
    upload_floorplan,
    download_floorplan,
    insert_model_2d,
    load_templates,
    load_section_from_page,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=jsonable_encoder(payload),
        status_code=status_code,
        media_type="application/json",
    )


def validate_required(params: dict, required_fields: list, endpoint: str, rid: str):
    missing = [f for f in required_fields if params.get(f) is None]
    if missing:
        log_json("WARNING", "VALIDATION_FAILED", request_id=rid, endpoint=endpoint,
                 missing_fields=missing)
        return False, respond_with_UI_payload(
            dict(error=f"Missing required fields: {', '.join(missing)}"),
            status_code=400
        )
    return True, None


def require_pool(pool, endpoint: str, rid: str):
    if pool is None:
        log_json("ERROR", "POOL_UNAVAILABLE", request_id=rid, endpoint=endpoint)
        return respond_with_UI_payload(
            dict(error="Database unavailable. Please try again later."),
            status_code=503
        )
    return None


def get_params(request_query_params, body):
    merged = dict(body) if body else {}
    merged.update(dict(request_query_params))
    return merged


def floorplan_to_walls(credentials, project_id, plan_id, user_id, page_number, mask=None, output_path=None):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_compute_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["wall_detector"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token

    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }

    payload = dict(
        project_id=project_id,
        plan_id=plan_id,
        user_id=user_id,
        page_number=page_number,
    )
    if mask:
        payload["mask"] = mask

    response = requests.post(
        f"{credentials['CloudRun']['APIs']['wall_detector']}/detect_wall",
        headers=headers,
        json=payload,
        timeout=300,
    )

    # Validate wall detector response
    if response.status_code != 200:
        log_json("ERROR", "WALL_DETECTOR_FAILED",
                 status_code=response.status_code,
                 page_number=page_number,
                 response_preview=response.text[:500])
        raise RuntimeError(f"Wall detector returned HTTP {response.status_code}")

    content_type = response.headers.get("content-type", "")
    if "image" not in content_type:
        log_json("ERROR", "WALL_DETECTOR_BAD_CONTENT_TYPE",
                 content_type=content_type,
                 page_number=page_number,
                 response_preview=response.text[:500])
        raise RuntimeError(f"Wall detector returned content-type '{content_type}', expected image")

    if len(response.content) < 1000:
        log_json("WARNING", "WALL_DETECTOR_SMALL_RESPONSE",
                 page_number=page_number,
                 content_length=len(response.content))

    if not output_path:
        output_path = Path("/tmp/floor_plan_wall_segmented.png")
    with open(output_path, "wb") as f:
        f.write(response.content)
    return Path(output_path)


def page_to_structured_2d(
    credentials,
    floor_plan_modeller_2d,
    project_id,
    plan_id,
    user_id,
    page_number,
    page_section_number,
    wall_segmented_path,
    floor_plan_processed_path,
    bounding_box_offset,
    transcription_block_with_centroids,
    transcription_headers_and_footers,
    floorplan_page_statistics,
    floorplan_baseline_page_source,
    verbose="False"
):
    """Process a single page section — kept from drywall_bq architecture."""
    floor_plan_modeller_2d.reload()
    wall_segmented_sectioned_path = load_section_from_page(
        wall_segmented_path,
        floor_plan_processed_path,
        bounding_box_offset,
        page_section_number
    )
    walls_2d, polygons, walls_2d_path, external_contour = floor_plan_modeller_2d.model(
        image_path=wall_segmented_sectioned_path,
        model_2d_path=f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(page_number).zfill(4)}.json",
        floor_plan_path=floor_plan_processed_path,
        transcription_block_with_centroids=transcription_block_with_centroids,
        transcription_headers_and_footers=transcription_headers_and_footers,
    )
    metadata = dict()
    if walls_2d and polygons:
        floor_plan_modeller_2d.load_drywall_choices(walls_2d, polygons)
        floor_plan_modeller_2d.load_ceiling_choices(polygons)

        floorplan_baseline, floorplan_page_statistics = FloorPlan2D.scale_to(
            floor_plan_path=floor_plan_processed_path
        )
        floorplan_baseline_page_source = upload_floorplan(
            floorplan_baseline, plan_id, project_id, credentials,
            index=str(page_number).zfill(4)
        )

        metadata = dict(
            size_in_bytes=floorplan_page_statistics["size"],
            height_in_pixels=floorplan_page_statistics["height_in_pixels"],
            width_in_pixels=floorplan_page_statistics["width_in_pixels"],
            height_in_points=floorplan_page_statistics["height_in_points"],
            width_in_points=floorplan_page_statistics["width_in_points"],
            origin=["LEFT", "TOP"],
            offset=(0, 0),
            contour_root_vertices=external_contour,
            scales_architectural=floor_plan_modeller_2d.scales_architectural,
            drywall_choices_color_codes=floor_plan_modeller_2d.drywall_choices_color_codes,
        )

    # --- DB write: insert_model_2d (now async PG, called via asyncio from sync thread) ---
    # Since page_to_structured_2d runs in ThreadPoolExecutor, we need to schedule
    # the async insert_model_2d on the event loop
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(insert_model_2d(
            dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata),
            floor_plan_modeller_2d.normalize_scale(floor_plan_modeller_2d.scale),
            page_number,
            page_section_number,
            plan_id,
            user_id,
            project_id,
            floorplan_baseline_page_source,
            None,  # pool not used — insert_model_2d uses fresh asyncpg.connect
            credentials,
        ))
    finally:
        loop.close()

    log_json("INFO", "PAGE_SECTION_COMPLETE",
             page_number=page_number,
             page_section_number=page_section_number,
             has_walls=walls_2d is not None,
             has_polygons=polygons is not None)


# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Floorplan-to-Structured-2D (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
HYPERPARAMETERS = load_hyperparameters()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singletons
pg_pool = None
DRYWALL_TEMPLATES = None


@app.on_event("startup")
async def startup():
    global pg_pool, DRYWALL_TEMPLATES

    # 1. PostgreSQL pool
    try:
        pg_pool = await create_pg_pool(CREDENTIALS)
        if pg_pool:
            log_json("INFO", "STARTUP_PG_SUCCESS", detail="PostgreSQL pool created")
        else:
            log_json("WARNING", "STARTUP_PG_DEGRADED", detail="PostgreSQL pool is None")
    except Exception as exc:
        log_json("ERROR", "STARTUP_PG_FAILED", error=f"{type(exc).__name__}: {exc}")
        pg_pool = None

    # 2. Load drywall templates (SKU data — rarely changes, cache at startup)
    if pg_pool:
        try:
            DRYWALL_TEMPLATES = await load_templates(pg_pool, CREDENTIALS)
            log_json("INFO", "STARTUP_TEMPLATES_LOADED", count=len(DRYWALL_TEMPLATES))
        except Exception as exc:
            log_json("ERROR", "STARTUP_TEMPLATES_FAILED", error=f"{type(exc).__name__}: {exc}")
            DRYWALL_TEMPLATES = []

    # 3. GCS client singleton
    get_gcs_client()

    log_json("INFO", "STARTUP_COMPLETE")


@app.on_event("shutdown")
async def shutdown():
    global pg_pool
    if pg_pool:
        await pg_pool.close()
        log_json("INFO", "SHUTDOWN", detail="PostgreSQL pool closed")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/floorplan_to_structured_2d")
async def floorplan_to_structured_2d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    enable_logging_on_stdout()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/floorplan_to_structured_2d")

    pool_err = require_pool(pg_pool, "/floorplan_to_structured_2d", rid)
    if pool_err:
        return pool_err

    params = get_params(request.query_params, None)
    try:
        body = await request.json()
        params = get_params(request.query_params, body)
    except Exception:
        pass

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id", "page_number"],
                                   "/floorplan_to_structured_2d", rid)
    if not valid:
        return err

    project_id = params["project_id"]
    user_id = params["user_id"]
    plan_id = params["plan_id"]
    page_number = params["page_number"]
    mask = params.get("mask")
    bounding_box_offsets = params.get("bounding_box_offsets")
    verbose = params.get("verbose", "False")
    page_number_padded = str(page_number).zfill(4)

    log_json("INFO", "REQUEST_PARAMS", request_id=rid,
             project_id=project_id, plan_id=plan_id, user_id=user_id, page_number=page_number)

    # --- Step 1: Download processed floorplan from GCS ---
    async with timed_step("download_floorplan", request_id=rid, page_number=page_number):
        floor_plan_processed_path = download_floorplan(
            user_id, plan_id, project_id, CREDENTIALS, page_number_padded
        )

    # --- Step 2: Parallel wall detection + transcription ---
    async with timed_step("parallel_wall_detect_and_transcribe", request_id=rid, page_number=page_number):
        futures = dict()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures["floorplan_to_walls"] = executor.submit(
                floorplan_to_walls,
                CREDENTIALS,
                project_id,
                plan_id,
                user_id,
                page_number,
                mask,
                output_path=f"/tmp/{project_id}/{plan_id}/{user_id}/floor_plan_wall_segmented_{page_number_padded}.png"
            )
            futures["transcriber"] = executor.submit(
                transcribe,
                CREDENTIALS,
                HYPERPARAMETERS,
                floor_plan_processed_path,
            )
        wall_segmented_path = futures["floorplan_to_walls"].result()
        transcription_block_with_centroids, transcription_headers_and_footers = futures["transcriber"].result()

    # --- Step 3: Upload wall segmented image ---
    async with timed_step("upload_wall_segmented", request_id=rid):
        upload_floorplan(wall_segmented_path, plan_id, project_id, CREDENTIALS, index=page_number_padded)

    log_json("INFO", "STEP_COMPLETE", request_id=rid, step="wall_detection_and_transcription",
             page_number=page_number)

    # --- Step 4: Use cached drywall templates ---
    templates = DRYWALL_TEMPLATES
    if not templates:
        log_json("WARNING", "TEMPLATES_CACHE_MISS", request_id=rid,
                 detail="Reloading templates from DB")
        templates = await load_templates(pg_pool, CREDENTIALS)

    # --- Step 5: Process page sections (kept from drywall_bq architecture) ---
    hyperparameters = HYPERPARAMETERS
    floorplan_baseline_page_source = None

    if not FloorPlan2D.is_none(wall_segmented_path):
        floorplan_baseline, floorplan_page_statistics = FloorPlan2D.scale_to(
            floor_plan_path=floor_plan_processed_path
        )
        floorplan_baseline_page_source = upload_floorplan(
            floorplan_baseline, plan_id, project_id, CREDENTIALS,
            index=page_number_padded
        )

        # Handle bounding_box_offsets (multi-section pages)
        if not bounding_box_offsets:
            bounding_box_offsets = [
                {"offset_top_left": (0, 0), "offset_bottom_right": (1, 1), "title": "I"}
            ]

        ip_address = request.headers.get("X-Client-IP", (request.client.host if request.client else None))

        async with timed_step("model_2d_generation", request_id=rid, page_number=page_number):
            vertex_ai_clients = FloorPlan2D.load_vertex_ai_clients(CREDENTIALS, ip_address, templates)
            futures = list()
            with ThreadPoolExecutor(max_workers=2) as executor:
                for bounding_box_offset in bounding_box_offsets:
                    log_json("INFO", "PROCESSING_SECTION",
                             request_id=rid,
                             page_number=page_number,
                             section=bounding_box_offset.get("title", "I"))
                    floor_plan_modeller_2d = FloorPlan2D(CREDENTIALS, hyperparameters, templates)
                    floor_plan_modeller_2d.from_vertex_ai_clients(*vertex_ai_clients)
                    futures.append(
                        executor.submit(
                            page_to_structured_2d,
                            CREDENTIALS,
                            floor_plan_modeller_2d,
                            project_id,
                            plan_id,
                            user_id,
                            page_number,
                            bounding_box_offset.get("title", "I"),
                            wall_segmented_path,
                            floor_plan_processed_path,
                            bounding_box_offset,
                            transcription_block_with_centroids,
                            transcription_headers_and_footers,
                            floorplan_page_statistics,
                            floorplan_baseline_page_source,
                            verbose,
                        )
                    )
                [future.result() for future in futures]
    else:
        log_json("WARNING", "WALL_SEGMENTATION_EMPTY", request_id=rid,
                 page_number=page_number,
                 detail="FloorPlan2D.is_none returned True — no walls detected")

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid,
             endpoint="/floorplan_to_structured_2d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             page_number=page_number,
             has_walls=not FloorPlan2D.is_none(wall_segmented_path) if 'wall_segmented_path' in dir() else False)
    return respond_with_UI_payload(dict(status="success", page_number=page_number))
