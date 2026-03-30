"""
main.py — drywall-takeoff-3d-fbm (drywall_bq repo)
Migrated from BigQuery to PostgreSQL.
All 15+ endpoints migrated. Keeps drywall_bq's page_section_number support.
"""

import os
import sys
import re
import uuid
import time as time_module
from datetime import timedelta, datetime, date, time
from decimal import Decimal
from base64 import b64encode
from pathlib import Path
import json
from time import time as from_unix_epoch
from collections import defaultdict
import asyncio
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pydantic_core import ValidationError
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import math

# --- Change 1: Import at top of file ---
# Change:
#   from preprocessing import preprocess
# To:
from preprocessing import preprocess, reprocess_pages_hires
from extrapolate_3d import Extrapolate3D
from helper import (
    create_pg_pool,
    pg_execute,
    pg_fetch_one,
    pg_fetch_all,
    log_json,
    timed_step,
    parse_jsonb,
    get_gcs_client,
    load_gcp_credentials,
    load_hyperparameters,
    enable_logging_on_stdout,
    sha256,
    upload_floorplan,
    insert_model_2d,
    is_duplicate,
    delete_plan,
    load_floorplan_to_structured_2d_ID_token,
    load_vertex_ai_client,
    classify_plan,
    load_templates,
)


# ---------------------------------------------------------------------------
# Validation & Response Helpers
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


# ---------------------------------------------------------------------------
# GCS Helper
# ---------------------------------------------------------------------------

def download_floorplan(plan_id, project_id, credentials, destination_path="/tmp/floor_plan.PDF"):
    client = get_gcs_client()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    blob.download_to_filename(destination_path)
    return f"gs://{credentials['CloudStorage']['bucket_name']}/{blob_path}"


def floorplan_to_structured_2d(credentials, id_token, project_id, plan_id, user_id, page_number, mask_factor=None, bounding_box_offsets=None):
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
    if mask_factor:
        payload["mask"] = mask_factor
    if bounding_box_offsets:
        payload["bounding_box_offsets"] = bounding_box_offsets
    try:
        requests.post(
            f"{credentials['CloudRun']['APIs']['floorplan_to_structured_2d']}/floorplan_to_structured_2d",
            headers=headers,
            json=payload,
            timeout=1800,
        )
    except Exception as e:
        log_json("WARNING", "FLOORPLAN_TO_2D_CALL_FAILED",
                 page_number=page_number, error=str(e))


# ---------------------------------------------------------------------------
# Database Operations (in main.py for Service 1 specific ops)
# ---------------------------------------------------------------------------

async def insert_project(payload_project, pool, credentials, rid=""):
    async with timed_step("insert_project", request_id=rid, project_id=payload_project.project_id):
        await pg_execute(
            pool,
            """
            INSERT INTO projects (
                project_id, project_name, project_location, fbm_branch,
                project_type, project_area, contractor_name, created_at, created_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), $8)
            ON CONFLICT (LOWER(project_id)) DO NOTHING
            """,
            [
                payload_project.project_id, payload_project.project_name,
                payload_project.project_location, payload_project.FBM_branch,
                payload_project.project_type, payload_project.project_area,
                payload_project.contractor_name, payload_project.created_by
            ],
            query_name="insert_project"
        )
        row = await pg_fetch_one(
            pool,
            "SELECT created_at FROM projects WHERE LOWER(project_id) = LOWER($1)",
            [payload_project.project_id],
            query_name="insert_project__get_created_at"
        )
        return row["created_at"].isoformat() if row else None


async def insert_plan(
    project_id, user_id, status, pool, credentials,
    payload_plan=None, plan_id=None, size_in_bytes=None,
    GCS_URL_floorplan=None, n_pages=None, sha_256=None
):
    if sha_256 is None:
        sha_256 = ''
    if plan_id and not sha_256:
        pdf_path = Path("/tmp/floor_plan.PDF")
        download_floorplan(plan_id, project_id, credentials, destination_path=pdf_path)
        sha_256 = sha256(pdf_path)
    if not plan_id:
        plan_id = payload_plan.plan_id
    plan_name, plan_type, file_type = '', '', ''
    if payload_plan:
        plan_name = payload_plan.plan_name
        plan_type = payload_plan.plan_type
        file_type = payload_plan.file_type
    if not n_pages:
        n_pages = 0
    if not GCS_URL_floorplan:
        GCS_URL_floorplan = ''
    if not size_in_bytes:
        size_in_bytes = 0

    await pg_execute(
        pool,
        """
        INSERT INTO plans (
            plan_id, project_id, user_id, status, plan_name, plan_type,
            file_type, pages, size_in_bytes, source, sha256, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW())
        ON CONFLICT (LOWER(project_id), LOWER(plan_id))
        DO UPDATE SET
            pages = EXCLUDED.pages,
            source = EXCLUDED.source,
            sha256 = EXCLUDED.sha256,
            status = EXCLUDED.status,
            size_in_bytes = EXCLUDED.size_in_bytes,
            user_id = EXCLUDED.user_id,
            updated_at = NOW()
        """,
        [plan_id, project_id, user_id, status, plan_name, plan_type,
         file_type, n_pages, size_in_bytes, GCS_URL_floorplan, sha_256],
        query_name="insert_plan"
    )


async def insert_model_2d_revision(
    model_2d, scale, page_number, plan_id, user_id, project_id, pool, credentials
):
    page_number = int(page_number)
    model_2d_json = json.dumps(model_2d)

    if not model_2d.get("metadata", None):
        row = await pg_fetch_one(
            pool,
            "SELECT model_2d->'metadata' AS metadata FROM models "
            "WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
            [project_id, plan_id, page_number],
            query_name="insert_model_2d_revision__fetch_metadata"
        )
        if row:
            metadata = parse_jsonb(row["metadata"])
            if metadata:
                model_2d["metadata"] = metadata
                model_2d_json = json.dumps(model_2d)

    row = await pg_fetch_one(
        pool,
        "SELECT MAX(revision_number) AS revision_number FROM model_revisions_2d "
        "WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, page_number],
        query_name="insert_model_2d_revision__max_rev"
    )
    revision_number = (row["revision_number"] or 0) + 1 if row else 1

    await pg_execute(
        pool,
        """
        INSERT INTO model_revisions_2d (
            plan_id, project_id, user_id, page_number, scale, model, created_at, revision_number
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW(), $7)
        """,
        [plan_id, project_id, user_id, page_number, scale or '', model_2d_json, revision_number],
        query_name="insert_model_2d_revision"
    )


async def insert_model_3d(
    model_3d, scale, page_number, plan_id, user_id, project_id, pool, credentials
):
    page_number = int(page_number)
    model_3d_json = json.dumps(model_3d)
    scale = scale or ''

    await pg_execute(
        pool,
        """
        UPDATE models SET
            model_3d = $1::jsonb,
            scale = CASE WHEN $2 = '' THEN scale ELSE $2 END,
            user_id = $3,
            updated_at = NOW()
        WHERE LOWER(project_id) = LOWER($4)
          AND LOWER(plan_id) = LOWER($5)
          AND page_number = $6
        """,
        [model_3d_json, scale, user_id, project_id, plan_id, page_number],
        query_name="insert_model_3d"
    )


async def insert_model_3d_revision(
    model_3d, scale, page_number, plan_id, user_id, project_id, pool, credentials
):
    page_number = int(page_number)
    model_3d_json = json.dumps(model_3d)

    row = await pg_fetch_one(
        pool,
        "SELECT MAX(revision_number) AS revision_number FROM model_revisions_3d "
        "WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, page_number],
        query_name="insert_model_3d_revision__max_rev"
    )
    revision_number = (row["revision_number"] or 0) + 1 if row else 1

    await pg_execute(
        pool,
        """
        INSERT INTO model_revisions_3d (
            plan_id, project_id, user_id, page_number, scale,
            model, takeoff, created_at, revision_number
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, '{}'::jsonb, NOW(), $7)
        """,
        [plan_id, project_id, user_id, page_number, scale or '', model_3d_json, revision_number],
        query_name="insert_model_3d_revision"
    )


async def insert_takeoff(
    takeoff, page_number, plan_id, user_id, project_id,
    revision_number, pool, credentials
):
    page_number = int(page_number)
    takeoff_json = json.dumps(takeoff)

    await pg_execute(
        pool,
        """
        UPDATE models SET takeoff = $1::jsonb, updated_at = NOW(), user_id = $2
        WHERE LOWER(project_id) = LOWER($3) AND LOWER(plan_id) = LOWER($4) AND page_number = $5
        """,
        [takeoff_json, user_id, project_id, plan_id, page_number],
        query_name="insert_takeoff__models"
    )

    if revision_number:
        revision_number = int(revision_number)
        await pg_execute(
            pool,
            """
            UPDATE model_revisions_3d SET takeoff = $1::jsonb, user_id = $2
            WHERE LOWER(project_id) = LOWER($3) AND LOWER(plan_id) = LOWER($4)
              AND page_number = $5 AND revision_number = $6
            """,
            [takeoff_json, user_id, project_id, plan_id, page_number, revision_number],
            query_name="insert_takeoff__revision_3d"
        )


async def delete_floorplan(project_id, plan_id, user_id, pool, credentials):
    params_2 = [project_id, plan_id]

    await pg_execute(pool,
        "DELETE FROM model_revisions_3d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        params_2, query_name="delete_floorplan__rev_3d")
    await pg_execute(pool,
        "DELETE FROM model_revisions_2d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        params_2, query_name="delete_floorplan__rev_2d")
    await pg_execute(pool,
        "DELETE FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        params_2, query_name="delete_floorplan__models")
    await pg_execute(pool,
        "DELETE FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND LOWER(user_id) = LOWER($3)",
        [project_id, plan_id, user_id], query_name="delete_floorplan__plans")

    client = get_gcs_client()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    prefix = f"{project_id.lower()}/{plan_id.lower()}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)
        log_json("INFO", "GCS_CLEANUP", prefix=prefix, blobs_deleted=len(blobs))


# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Drywall Takeoff (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
HYPERPARAMETERS = load_hyperparameters()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pg_pool = None
VERTEX_AI_CLIENT = None
VERTEX_AI_GENERATION_CONFIG = None
VERTEX_AI_MAX_RETRY = None


@app.on_event("startup")
async def startup():
    global pg_pool, VERTEX_AI_CLIENT, VERTEX_AI_GENERATION_CONFIG, VERTEX_AI_MAX_RETRY

    try:
        pg_pool = await create_pg_pool(CREDENTIALS)
        if pg_pool:
            log_json("INFO", "STARTUP_PG_SUCCESS", detail="PostgreSQL pool created")
        else:
            log_json("WARNING", "STARTUP_PG_DEGRADED", detail="PostgreSQL pool is None")
    except Exception as exc:
        log_json("ERROR", "STARTUP_PG_FAILED", error=f"{type(exc).__name__}: {exc}")
        pg_pool = None

    try:
        VERTEX_AI_CLIENT, VERTEX_AI_GENERATION_CONFIG = load_vertex_ai_client(CREDENTIALS)
        VERTEX_AI_MAX_RETRY = CREDENTIALS["VertexAI"]["llm"]["max_retry"]
        log_json("INFO", "STARTUP_VERTEX_AI_SUCCESS",
                 model=CREDENTIALS["VertexAI"]["llm"]["model_name"])
    except Exception as exc:
        log_json("ERROR", "STARTUP_VERTEX_AI_FAILED", error=f"{type(exc).__name__}: {exc}")

    get_gcs_client()
    log_json("INFO", "STARTUP_COMPLETE")


@app.on_event("shutdown")
async def shutdown():
    global pg_pool
    if pg_pool:
        await pg_pool.close()
        log_json("INFO", "SHUTDOWN", detail="PostgreSQL pool closed")


class PayloadProject(BaseModel):
    project_id: str
    project_name: str
    project_location: str
    project_area: str
    project_type: str
    contractor_name: str
    FBM_branch: str
    created_by: str


class PayloadPlan(BaseModel):
    plan_id: str
    plan_name: str
    plan_type: str
    file_type: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate_project")
async def generate_project(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/generate_project")
    pool_err = require_pool(pg_pool, "/generate_project", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    try: payload_project = PayloadProject(**params if params else body)
    except (ValidationError, Exception):
        try: payload_project = PayloadProject(**body)
        except (ValidationError, Exception) as e:
            return respond_with_UI_payload(dict(error=f"Invalid project payload: {e}"), status_code=400)

    created_at = await insert_project(payload_project, pg_pool, CREDENTIALS, rid=rid)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/generate_project",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(project_id=payload_project.project_id, project_name=payload_project.project_name, created_at=created_at))


@app.post("/load_projects")
async def load_projects(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_projects")
    pool_err = require_pool(pg_pool, "/load_projects", rid)
    if pool_err: return pool_err

    rows = await pg_fetch_all(pg_pool, "SELECT * FROM projects", query_name="load_projects")
    projects = [dict(row) for row in rows]

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_projects",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             project_count=len(projects))
    return respond_with_UI_payload(jsonable_encoder({"projects": projects}))


@app.post("/load_project_plans")
async def load_project_plans(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_project_plans")
    pool_err = require_pool(pg_pool, "/load_project_plans", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id"], "/load_project_plans", rid)
    if not valid: return err

    project_row = await pg_fetch_one(pg_pool, "SELECT * FROM projects WHERE LOWER(project_id) = LOWER($1)",
                                     [params["project_id"]], query_name="load_project_plans__project")
    if not project_row:
        return respond_with_UI_payload(dict(project_metadata=dict(), project_plans=list()))

    plan_rows = await pg_fetch_all(pg_pool, "SELECT * FROM plans WHERE LOWER(project_id) = LOWER($1)",
                                   [params["project_id"]], query_name="load_project_plans__plans")

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_project_plans",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(jsonable_encoder({
        "project_metadata": dict(project_row),
        "project_plans": [dict(row) for row in plan_rows]
    }))


@app.post("/generate_floorplan_upload_signed_URL")
async def generate_floorplan_upload_signed_URL(request: Request) -> str:
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/generate_floorplan_upload_signed_URL")
    pool_err = require_pool(pg_pool, "/generate_floorplan_upload_signed_URL", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "user_id", "plan"], "/generate_floorplan_upload_signed_URL", rid)
    if not valid: return err

    project_id = params["project_id"]
    user_id = params["user_id"]
    payload_plan = PayloadPlan(**params["plan"])

    await insert_plan(project_id, user_id, "NOT STARTED", pg_pool, CREDENTIALS, payload_plan=payload_plan)

    client = get_gcs_client()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{payload_plan.plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="PUT",
        content_type="application/octet-stream",
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/generate_floorplan_upload_signed_URL",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return url


@app.post("/load_plan_pages")
async def load_plan_pages(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_plan_pages")
    pool_err = require_pool(pg_pool, "/load_plan_pages", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id"], "/load_plan_pages", rid)
    if not valid: return err

    rows = await pg_fetch_all(
        pg_pool, "SELECT * FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        [params["project_id"], params["plan_id"]], query_name="load_plan_pages"
    )
    records = [dict(row) for row in rows]
    plan_metadata = records[0] if records else dict()

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_plan_pages",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(plan_metadata=plan_metadata, plan_pages=records))


@app.post("/floorplan_to_2d")
async def floorplan_to_2d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/floorplan_to_2d")
    pool_err = require_pool(pg_pool, "/floorplan_to_2d", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id"], "/floorplan_to_2d", rid)
    if not valid: return err

    project_id = params["project_id"]
    user_id = params["user_id"]
    plan_id = params["plan_id"]

    pdf_path = Path("/tmp/floor_plan.PDF")
    async with timed_step("download_floorplan", request_id=rid, plan_id=plan_id):
        GCS_URL_floorplan = download_floorplan(plan_id, project_id, CREDENTIALS, destination_path=pdf_path)

    sha_256 = sha256(pdf_path)

    async with timed_step("is_duplicate_check", request_id=rid, plan_id=plan_id):
        plan_duplicate = await is_duplicate(pg_pool, CREDENTIALS, pdf_path, project_id)
    if plan_duplicate:
        await delete_plan(pg_pool, CREDENTIALS, plan_id, project_id)
        log_json("WARNING", "DUPLICATE_PLAN", request_id=rid, plan_id=plan_id)
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))

    client = get_gcs_client()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"tmp/{user_id.lower()}/{project_id.lower()}/{plan_id.lower()}/floorplan_structured_2d.json"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.delete()

    size_in_bytes = Path(pdf_path).stat().st_size

    async with timed_step("preprocess_pdf", request_id=rid, plan_id=plan_id, volume_context={"file_size_bytes": size_in_bytes}):
        floor_plan_paths_vector, floor_plan_paths_preprocessed = preprocess(pdf_path)

    await insert_plan(
        project_id, user_id, "IN PROGRESS", pg_pool, CREDENTIALS,
        plan_id=plan_id, size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
        sha_256=sha_256,
    )

    log_json("INFO", "STEP_COMPLETE", request_id=rid, step="insert_plan_in_progress",
             volume_context={"n_pages": len(floor_plan_paths_preprocessed)})

    walls_2d_all = dict(pages=list())
    status = "COMPLETED"
    vertex_ai_client_parameters = (VERTEX_AI_CLIENT, VERTEX_AI_GENERATION_CONFIG, VERTEX_AI_MAX_RETRY)

    def process_single_page(index, floor_plan_vector, floor_plan_path, creds):
        p_type = classify_plan(floor_plan_path, vertex_ai_client_parameters)
        baseline_src = upload_floorplan(floor_plan_vector, plan_id, project_id, creds, index=str(index).zfill(2))
        page_src = upload_floorplan(floor_plan_path, plan_id, project_id, creds, index=str(index).zfill(2))
        return p_type, baseline_src, page_src

    try:
        id_token = load_floorplan_to_structured_2d_ID_token(CREDENTIALS)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = list()
            for index, (floor_plan_vector, floor_plan_path) in enumerate(zip(floor_plan_paths_vector, floor_plan_paths_preprocessed)):
                futures.append(executor.submit(process_single_page, index, floor_plan_vector, floor_plan_path, CREDENTIALS))
            results = [future.result() for future in futures]

            plan_types = [r[0] for r in results]
            floorplan_baseline_page_sources = [r[1] for r in results]
            floorplan_page_sources = [r[2] for r in results]

            floor_indices = [i for i, pt in enumerate(plan_types) if pt["plan_type"].upper().find("FLOOR") != -1]

            log_json("INFO", "CLASSIFICATION_COMPLETE", request_id=rid,
                     total_pages=len(plan_types),
                     floor_plan_pages=floor_indices,
                     all_types=[pt["plan_type"] for pt in plan_types])

            # Re-convert FLOOR pages at high-res DPI=400
            if floor_indices:
                reprocess_pages_hires(pdf_path, floor_indices)
                # Re-upload high-res versions for FLOOR pages
                for fi in floor_indices:
                    floorplan_page_sources[fi] = upload_floorplan(
                        floor_plan_paths_preprocessed[fi], plan_id, project_id, CREDENTIALS,
                        index=str(fi).zfill(2)
                    )

            # Dispatch Service 2 calls in parallel
            for index, plan_type in enumerate(plan_types):
                if plan_type["plan_type"].upper().find("FLOOR") == -1:
                    continue

                mask_factor = plan_type.get("mask_factor")
                bounding_box_offsets = plan_type.get("bounding_box_offsets")

                executor.submit(
                    floorplan_to_structured_2d,
                    CREDENTIALS, id_token, project_id, plan_id, user_id,
                    str(index).zfill(2), mask_factor, bounding_box_offsets
                )

            # Poll for results
            for page_number, (plan_type, _, floorplan_page_source) in enumerate(zip(plan_types, floorplan_baseline_page_sources, floorplan_page_sources)):
                if plan_type["plan_type"].upper().find("FLOOR") == -1:
                    continue
                timeout = from_unix_epoch() + 3600
                poll_count = 0
                while from_unix_epoch() < timeout:
                    query_output = await pg_fetch_all(
                        pg_pool,
                        "SELECT scale, model_2d FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
                        [project_id, plan_id, page_number],
                        query_name="floorplan_to_2d__poll_model"
                    )
                    poll_count += 1
                    if query_output:
                        break
                    await asyncio.sleep(2)

                log_json("INFO", "POLL_COMPLETE", request_id=rid, step="poll_2d_model",
                         page_number=page_number, poll_iterations=poll_count)

                model_2d_raw = query_output[0]["model_2d"] if query_output else None
                walls_2d = parse_jsonb(model_2d_raw) if model_2d_raw else None
                if not walls_2d or not walls_2d.get("polygons") or not walls_2d.get("walls_2d"):
                    await pg_execute(
                        pg_pool,
                        "DELETE FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
                        [project_id, plan_id, page_number],
                        query_name="floorplan_to_2d__delete_empty_model"
                    )
                    continue
                await pg_execute(
                    pg_pool,
                    "UPDATE models SET source = $1 WHERE LOWER(project_id) = LOWER($2) AND LOWER(plan_id) = LOWER($3) AND page_number = $4",
                    [floorplan_page_source, project_id, plan_id, page_number],
                    query_name="floorplan_to_2d__update_source"
                )
                page = dict(
                    plan_id=plan_id,
                    page_number=page_number,
                    page_type=plan_type["plan_type"].upper(),
                    scale=query_output[0]["scale"],
                    walls_2d=walls_2d["walls_2d"],
                    polygons=walls_2d["polygons"],
                    **walls_2d.get("metadata", dict())
                )
                walls_2d_all["pages"].append(page)

    except Exception as e:
        log_json("ERROR", "FLOORPLAN_EXTRACTION_FAILED", request_id=rid, error=str(e))
        status = "FAILED"

    await insert_plan(
        project_id, user_id, status, pg_pool, CREDENTIALS,
        plan_id=plan_id, size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
        sha_256=sha_256,
    )

    with open("/tmp/floorplan_structured_2d.json", 'w') as f:
        json.dump(walls_2d_all, f, indent=4)
    blob.upload_from_filename("/tmp/floorplan_structured_2d.json")

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/floorplan_to_2d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             status=status, page_count=len(walls_2d_all["pages"]))
    return respond_with_UI_payload(walls_2d_all)


@app.post("/update_floorplan_to_2d")
async def update_floorplan_to_2d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/update_floorplan_to_2d")
    pool_err = require_pool(pg_pool, "/update_floorplan_to_2d", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id", "page_number"], "/update_floorplan_to_2d", rid)
    if not valid: return err

    await insert_model_2d(
        dict(walls_2d=params.get("walls_2d"), polygons=params.get("polygons")),
        params.get("scale"), params["page_number"], params["plan_id"], params["user_id"],
        params["project_id"], None, None, pg_pool, CREDENTIALS
    )
    await insert_model_2d_revision(
        dict(walls_2d=params.get("walls_2d"), polygons=params.get("polygons")),
        params.get("scale"), params["page_number"], params["plan_id"], params["user_id"],
        params["project_id"], pg_pool, CREDENTIALS
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/update_floorplan_to_2d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(status="success"))


@app.post("/update_scale")
async def update_scale(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/update_scale")
    pool_err = require_pool(pg_pool, "/update_scale", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["scale", "project_id", "plan_id", "page_number"], "/update_scale", rid)
    if not valid: return err

    await pg_execute(
        pg_pool,
        "UPDATE models SET scale = $1 WHERE LOWER(project_id) = LOWER($2) AND LOWER(plan_id) = LOWER($3) AND page_number = $4",
        [params["scale"], params["project_id"], params["plan_id"], int(params["page_number"])],
        query_name="update_scale"
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/update_scale",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(status="success"))


@app.post("/load_scale")
async def load_scale(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_scale")
    pool_err = require_pool(pg_pool, "/load_scale", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number"], "/load_scale", rid)
    if not valid: return err

    row = await pg_fetch_one(
        pg_pool,
        "SELECT scale FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [params["project_id"], params["plan_id"], int(params["page_number"])],
        query_name="load_scale"
    )
    scale = row["scale"] if row else None

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_scale",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(scale=scale))


@app.post("/floorplan_to_3d")
async def floorplan_to_3d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/floorplan_to_3d")
    pool_err = require_pool(pg_pool, "/floorplan_to_3d", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id", "page_number"], "/floorplan_to_3d", rid)
    if not valid: return err

    project_id = params["project_id"]
    user_id = params["user_id"]
    plan_id = params["plan_id"]
    index = params["page_number"]
    index_int = int(index)

    row = await pg_fetch_one(
        pg_pool,
        "SELECT scale, model_2d FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, index_int],
        query_name="floorplan_to_3d__get_model"
    )
    if not row:
        return respond_with_UI_payload(dict(error="Model not found"), status_code=404)

    scale = row["scale"]
    model_2d = parse_jsonb(row["model_2d"])

    model_2d_path = f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(index).zfill(2)}.json"
    Path(model_2d_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_2d_path, 'w') as f:
        json.dump([model_2d.get("walls_2d", []), model_2d.get("polygons", [])], f)

    polygons_path = f"/tmp/{project_id}/{plan_id}/{user_id}/polygons_{str(index).zfill(2)}.json"
    with open(polygons_path, 'w') as f:
        json.dump(model_2d.get("polygons", []), f)

    floor_plan_modeller_3d = Extrapolate3D(HYPERPARAMETERS)
    walls_3d, polygons_3d, walls_3d_path, polygons_3d_path = floor_plan_modeller_3d.extrapolate(
        scale, model_2d_path=model_2d_path, polygons_path=polygons_path
    )
    walls_3d, polygons_3d = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d, polygons_3d)
    gltf_paths = floor_plan_modeller_3d.gltf(model_2d_path=model_2d_path, polygons_path=polygons_path)
    model_3d_path = floor_plan_modeller_3d.save_plot_3d(walls_3d_path, polygons_3d_path)

    metadata = parse_jsonb(model_2d.get("metadata")) if model_2d else None

    async with timed_step("upload_3d_assets", request_id=rid):
        upload_floorplan(model_3d_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
        for gltf_path in gltf_paths:
            upload_floorplan(gltf_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2), directory="gltf")

    await insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/floorplan_to_3d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(walls_3d=walls_3d, polygons=polygons_3d, metadata=metadata))


@app.post("/update_floorplan_to_3d")
async def update_floorplan_to_3d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/update_floorplan_to_3d")
    pool_err = require_pool(pg_pool, "/update_floorplan_to_3d", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id", "page_number"], "/update_floorplan_to_3d", rid)
    if not valid: return err

    await insert_model_3d(
        dict(walls_3d=params.get("walls_3d"), polygons=params.get("polygons")),
        params.get("scale"), params["page_number"], params["plan_id"],
        params["user_id"], params["project_id"], pg_pool, CREDENTIALS
    )
    await insert_model_3d_revision(
        dict(walls_3d=params.get("walls_3d"), polygons=params.get("polygons")),
        params.get("scale"), params["page_number"], params["plan_id"],
        params["user_id"], params["project_id"], pg_pool, CREDENTIALS
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/update_floorplan_to_3d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(status="success"))


@app.post("/load_3d_all")
async def load_3d_all(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_3d_all")
    pool_err = require_pool(pg_pool, "/load_3d_all", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id"], "/load_3d_all", rid)
    if not valid: return err

    rows = await pg_fetch_all(
        pg_pool,
        "SELECT page_number, scale, model_3d, model_2d->'metadata' AS metadata FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        [params["project_id"], params["plan_id"]],
        query_name="load_3d_all"
    )

    walls_3d_all = dict(pages=list())
    for row in rows:
        if not row["model_3d"]: continue
        model_3d = parse_jsonb(row["model_3d"])
        if not model_3d: continue
        metadata = parse_jsonb(row["metadata"]) or {}
        page = dict(
            plan_id=params["plan_id"],
            page_number=row["page_number"],
            walls_3d=model_3d.get("walls_3d", []),
            polygons=model_3d.get("polygons", []),
            scale=row["scale"],
            **metadata,
        )
        walls_3d_all["pages"].append(page)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_3d_all",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2),
             page_count=len(walls_3d_all["pages"]))
    return respond_with_UI_payload(walls_3d_all)


@app.post("/load_2d_revision")
async def load_2d_revision(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_2d_revision")
    pool_err = require_pool(pg_pool, "/load_2d_revision", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number", "revision_number"], "/load_2d_revision", rid)
    if not valid: return err

    row = await pg_fetch_one(
        pg_pool,
        "SELECT model FROM model_revisions_2d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3 AND revision_number = $4",
        [params["project_id"], params["plan_id"], int(params["page_number"]), int(params["revision_number"])],
        query_name="load_2d_revision"
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_2d_revision",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(parse_jsonb(row["model"]) if row else dict())


@app.post("/load_3d_revision")
async def load_3d_revision(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_3d_revision")
    pool_err = require_pool(pg_pool, "/load_3d_revision", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number", "revision_number"], "/load_3d_revision", rid)
    if not valid: return err

    row = await pg_fetch_one(
        pg_pool,
        "SELECT model FROM model_revisions_3d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3 AND revision_number = $4",
        [params["project_id"], params["plan_id"], int(params["page_number"]), int(params["revision_number"])],
        query_name="load_3d_revision"
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_3d_revision",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(parse_jsonb(row["model"]) if row else dict())


@app.post("/load_available_revision_numbers_3d")
async def load_available_revision_numbers_3d(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/load_available_revision_numbers_3d")
    pool_err = require_pool(pg_pool, "/load_available_revision_numbers_3d", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number"], "/load_available_revision_numbers_3d", rid)
    if not valid: return err

    rows = await pg_fetch_all(
        pg_pool,
        "SELECT revision_number FROM model_revisions_3d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [params["project_id"], params["plan_id"], int(params["page_number"])],
        query_name="load_available_revision_numbers_3d"
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/load_available_revision_numbers_3d",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload([row["revision_number"] for row in rows if row["revision_number"] is not None])


@app.post("/generate_drywall_overlaid_floorplan_download_signed_URL")
async def generate_drywall_overlaid_floorplan_download_signed_URL(request: Request) -> str:
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/generate_drywall_overlaid_floorplan_download_signed_URL")
    pool_err = require_pool(pg_pool, "/generate_drywall_overlaid_floorplan_download_signed_URL", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id", "page_number"], "/generate_drywall_overlaid_floorplan_download_signed_URL", rid)
    if not valid: return err

    project_id = params["project_id"]
    plan_id = params["plan_id"]
    index = int(params["page_number"])

    # Poll for completion
    plan_row = await pg_fetch_one(pg_pool, "SELECT pages FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
                                  [project_id, plan_id], query_name="signed_url__get_pages")
    if not plan_row:
        return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=404)

    n_pages = plan_row["pages"] or 1
    timeout = from_unix_epoch() + (n_pages * 120)
    status = "IN PROGRESS"
    while from_unix_epoch() < timeout:
        row = await pg_fetch_one(pg_pool, "SELECT status FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
                                 [project_id, plan_id], query_name="signed_url__poll_status")
        if row and row["status"] == "COMPLETED":
            status = "COMPLETED"
            break
        await asyncio.sleep(2)

    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed"), status_code=500)

    row = await pg_fetch_one(
        pg_pool,
        "SELECT target_drywalls FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, index],
        query_name="signed_url__get_target_drywalls"
    )
    if not row or not row["target_drywalls"]:
        return respond_with_UI_payload(dict(error="Target drywalls not found"), status_code=404)

    blob_path = row["target_drywalls"]
    if blob_path.startswith("gs://"):
        _, _, _, blob_path = blob_path.split('/', 3)

    client = get_gcs_client()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="GET",
    )

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/generate_drywall_overlaid_floorplan_download_signed_URL",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return url


@app.post("/remove_floorplan")
async def remove_floorplan(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/remove_floorplan")
    pool_err = require_pool(pg_pool, "/remove_floorplan", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "user_id", "plan_id"], "/remove_floorplan", rid)
    if not valid: return err

    await delete_floorplan(params["project_id"], params["plan_id"], params["user_id"], pg_pool, CREDENTIALS)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/remove_floorplan",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(dict(status="success"))


@app.post("/compute_takeoff")
async def compute_takeoff(request: Request):
    rid = str(uuid.uuid4())[:8]
    request_start = time_module.perf_counter()
    log_json("INFO", "REQUEST_START", request_id=rid, endpoint="/compute_takeoff")
    pool_err = require_pool(pg_pool, "/compute_takeoff", rid)
    if pool_err: return pool_err

    params = get_params(request.query_params, None)
    try: body = await request.json()
    except Exception: body = dict()
    params = get_params(request.query_params, body)

    valid, err = validate_required(params, ["project_id", "plan_id", "user_id", "page_number"], "/compute_takeoff", rid)
    if not valid: return err

    walls_3d_JSON = params.get("walls_3d", list())
    polygons_JSON = params.get("polygons", list())
    index = params["page_number"]
    index_int = int(index)
    project_id = params["project_id"]
    plan_id = params["plan_id"]
    user_id = params["user_id"]
    revision_number = params.get("revision_number", '')

    row = await pg_fetch_one(
        pg_pool,
        "SELECT scale FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
        [project_id, plan_id, index_int],
        query_name="compute_takeoff__get_scale"
    )
    scale = row["scale"] if row else None

    DRYWALL_TEMPLATES = await load_templates(pg_pool, CREDENTIALS)

    async with timed_step("download_floorplan", request_id=rid, plan_id=plan_id):
        pdf_path = Path("/tmp/floor_plan.PDF")
        download_floorplan(plan_id, project_id, CREDENTIALS, destination_path=pdf_path)

    if not walls_3d_JSON:
        if revision_number:
            row = await pg_fetch_one(
                pg_pool,
                "SELECT model FROM model_revisions_3d WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3 AND revision_number = $4",
                [project_id, plan_id, index_int, int(revision_number)],
                query_name="compute_takeoff__get_3d_revision"
            )
            walls_3d_JSON = parse_jsonb(row["model"]) if row else list()
        else:
            row = await pg_fetch_one(
                pg_pool,
                "SELECT model_3d FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
                [project_id, plan_id, index_int],
                query_name="compute_takeoff__get_3d"
            )
            walls_3d_JSON = parse_jsonb(row["model_3d"]) if row else list()
        if walls_3d_JSON is None:
            walls_3d_JSON = list()

    floor_plan_modeller_3d = Extrapolate3D(HYPERPARAMETERS)
    if scale and scale != "1/4``=1`0``":
        pixel_aspect_ratio_new = floor_plan_modeller_3d.compute_pixel_aspect_ratio(scale, HYPERPARAMETERS["pixel_aspect_ratio_to_feet"])
        walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.recompute_dimensions_walls_and_polygons(walls_3d_JSON, polygons_JSON, pixel_aspect_ratio_new, pdf_path)
    walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d_JSON, polygons_JSON)

    drywall_takeoff = dict(
        total=dict(roof=0, wall=0),
        per_drywall=dict(
            roof=defaultdict(lambda: defaultdict(lambda: 0)),
            wall=defaultdict(lambda: defaultdict(lambda: 0))
        )
    )
    for wall in walls_3d_JSON:
        surface_area = wall["height"] * wall["length"]
        drywall_count = 0
        for drywall in wall["surfaces_drywall"]:
            if not drywall["enabled"]:
                continue
            if drywall.get("type_stacked"):
                stack_length = len(drywall["type_stacked"])
                for stack_index in range(stack_length):
                    drywall_takeoff["total"]["wall"] += surface_area / stack_length
                    drywall_takeoff["per_drywall"]["wall"][drywall["type_stacked"][stack_index]]["area"] += surface_area / stack_length
            else:
                drywall_takeoff["total"]["wall"] += surface_area
                drywall_takeoff["per_drywall"]["wall"][drywall["type"]]["area"] += surface_area
            drywall_count += 1

    for polygon in polygons_JSON:
        if polygon.get("polygon_drywall", {}).get("enabled"):
            ceiling_area = polygon.get("area", 0)
            ceiling_type = polygon["polygon_drywall"]["type"]
            drywall_takeoff["total"]["roof"] += ceiling_area
            drywall_takeoff["per_drywall"]["roof"][ceiling_type]["area"] += ceiling_area

    takeoff = json.loads(json.dumps(drywall_takeoff))
    await insert_takeoff(takeoff, index, plan_id, user_id, project_id, revision_number, pg_pool, CREDENTIALS)

    log_json("INFO", "REQUEST_COMPLETE", request_id=rid, endpoint="/compute_takeoff",
             total_duration_ms=round((time_module.perf_counter() - request_start) * 1000, 2))
    return respond_with_UI_payload(takeoff)
