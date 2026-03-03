import os
import sys
import re
import logging
from datetime import timedelta, datetime, date, time
from decimal import Decimal
from base64 import b64encode
from ruamel.yaml import YAML
from pathlib import Path
import json
from time import time as from_unix_epoch
from time import sleep
from collections import defaultdict
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pydantic_core import ValidationError
from concurrent.futures import ThreadPoolExecutor

from google.cloud.storage import Client as CloudStorageClient
from google.cloud import bigquery
from google.cloud import secretmanager

import pandas as pd
import numpy as np
import math

from preprocessing import preprocess
from extrapolate_3d import Extrapolate3D
from helper import (
    load_bigquery_client,
    bigquery_run,
    sha256,
    upload_floorplan,
    insert_model_2d,
    is_duplicate,
    delete_plan,
    load_floorplan_to_structured_2d_ID_token,
    load_vertex_ai_client,
    classify_plan,
)


def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json",
    )


def download_floorplan(plan_id, project_id, credentials, destination_path="/tmp/floor_plan.PDF"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)

    blob.download_to_filename(destination_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"


def insert_model_2d_revision(
    model_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    bigquery_client,
    credentials
    ):
    if not model_2d.get("metadata", None):
        GBQ_query = f"SELECT model_2d.metadata FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
        query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
        metadata = list(query_output)[0].metadata
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        model_2d["metadata"] = metadata
    GBQ_query = """
    SELECT MAX(revision_number) AS revision_number FROM `drywall_takeoff.model_revisions_2d` WHERE 
    LOWER(project_id) = LOWER(@project_id) AND LOWER(plan_id) = LOWER(@plan_id) AND page_number = @page_number;
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number)
        ]
    )
    query_output = list(bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result())
    
    if query_output and query_output[0].revision_number is not None:
        revision_number = query_output[0].revision_number + 1
    else:
        revision_number = 1

    GBQ_query = """
    INSERT INTO `drywall_takeoff.model_revisions_2d` (
        plan_id,
        project_id,
        user_id,
        page_number,
        scale,
        model,
        created_at,
        revision_number
    )
    VALUES (
        @plan_id,
        @project_id,
        @user_id,
        @page_number,
        @scale,
        @model_2d,
        CURRENT_TIMESTAMP(),
        @revision_number
    );
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("scale", "STRING", scale),
            bigquery.ScalarQueryParameter("model_2d", "JSON", model_2d),
            bigquery.ScalarQueryParameter("revision_number", "INT64", revision_number)
        ]
    )

    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output


def insert_model_3d_revision(
    model_3d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    bigquery_client,
    credentials
    ):
    GBQ_query = """
    SELECT MAX(revision_number) AS revision_number FROM `drywall_takeoff.model_revisions_3d` WHERE 
    LOWER(project_id) = LOWER(@project_id) AND LOWER(plan_id) = LOWER(@plan_id) AND page_number = @page_number;
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number)
        ]
    )
    query_output = list(bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result())
    
    if query_output and query_output[0].revision_number is not None:
        revision_number = query_output[0].revision_number + 1
    else:
        revision_number = 1

    GBQ_query = """
    INSERT INTO `drywall_takeoff.model_revisions_3d` (
        plan_id,
        project_id,
        user_id,
        page_number,
        scale,
        model,
        takeoff,
        created_at,
        revision_number
    )
    VALUES (
        @plan_id,
        @project_id,
        @user_id,
        @page_number,
        @scale,
        @model_3d,
        JSON '{}',
        CURRENT_TIMESTAMP(),
        @revision_number
    );
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("scale", "STRING", scale),
            bigquery.ScalarQueryParameter("model_3d", "JSON", model_3d),
            bigquery.ScalarQueryParameter("revision_number", "INT64", revision_number)
        ]
    )

    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output


def insert_model_3d(
    model_3d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    bigquery_client,
    credentials
    ):
    GBQ_query = """
    UPDATE `drywall_takeoff.models` as t
    SET
        model_3d = @model_3d,
        scale = COALESCE(NULLIF(@scale, ''), t.scale),
        user_id = @user_id,
        updated_at = CURRENT_TIMESTAMP()
    WHERE
        LOWER(project_id) = LOWER(@project_id)
        AND LOWER(plan_id) = LOWER(@plan_id)
        AND page_number = @page_number
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("scale", "STRING", scale),
            bigquery.ScalarQueryParameter("model_3d", "JSON", model_3d)
        ]
    )
    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output


def delete_floorplan(project_id, plan_id, user_id, bigquery_client, credentials):
    GBQ_query = """
    DELETE FROM `drywall_takeoff.plans`
    WHERE
        LOWER(project_id) = LOWER(@project_id)
        AND LOWER(plan_id) = LOWER(@plan_id)
        AND LOWER(user_id) = LOWER(@user_id);
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()

    GBQ_query = """
    DELETE FROM `drywall_takeoff.models`
    WHERE
        LOWER(project_id) = LOWER(@project_id)
        AND LOWER(plan_id) = LOWER(@plan_id)
        AND LOWER(user_id) = LOWER(@user_id);
    """
    bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()

    GBQ_query = """
    DELETE FROM `drywall_takeoff.model_revisions_2d`
    WHERE
        LOWER(project_id) = LOWER(@project_id)
        AND LOWER(plan_id) = LOWER(@plan_id)
        AND LOWER(user_id) = LOWER(@user_id);
    """
    bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()

    GBQ_query = """
    DELETE FROM `drywall_takeoff.model_revisions_3d`
    WHERE
        LOWER(project_id) = LOWER(@project_id)
        AND LOWER(plan_id) = LOWER(@plan_id)
        AND LOWER(user_id) = LOWER(@user_id);
    """
    bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()

    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    prefix = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)


def insert_takeoff(
    takeoff,
    page_number,
    plan_id,
    user_id,
    project_id,
    revision_number,
    bigquery_client,
    credentials
    ):
    GBQ_query = """
    UPDATE `drywall_takeoff.models` t
    SET
        takeoff = @takeoff,
        updated_at = CURRENT_TIMESTAMP(),
        user_id = @user_id
    WHERE
        LOWER(project_id) = LOWER(@project_id)
        AND LOWER(plan_id) = LOWER(@plan_id)
        AND page_number = @page_number
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            bigquery.ScalarQueryParameter("takeoff", "JSON", takeoff)
        ]
    )
    query_output_takeoff_insert = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()

    if revision_number:
        GBQ_query = """
        UPDATE `drywall_takeoff.model_revisions_3d` t
        SET
            takeoff = @takeoff,
            user_id = @user_id
        WHERE
            LOWER(project_id) = LOWER(@project_id)
            AND LOWER(plan_id) = LOWER(@plan_id)
            AND page_number = @page_number
            AND revision_number = @revision_number
        """
        job_config = dict(
            query_parameters=[
                bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
                bigquery.ScalarQueryParameter("takeoff", "JSON", takeoff),
                bigquery.ScalarQueryParameter("revision_number", "INT64", revision_number)
            ]
        )
        bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()

    return query_output_takeoff_insert


def insert_plan(
    project_id,
    user_id,
    status,
    bigquery_client,
    credentials,
    payload_plan=None,
    plan_id=None,
    size_in_bytes=None,
    GCS_URL_floorplan=None,
    n_pages=None
    ):
    GBQ_query = """
    MERGE `drywall_takeoff.plans` t
    USING (
        SELECT
            @plan_id AS plan_id,
            @project_id AS project_id,
            @user_id AS user_id,
            @status AS status,
            @plan_name AS plan_name,
            @plan_type AS plan_type,
            @file_type AS file_type,
            @pages AS pages,
            @size_in_bytes AS size_in_bytes,
            @source AS source,
            @sha256 AS sha256
    ) s
    ON LOWER(t.project_id) = LOWER(s.project_id) AND LOWER(t.plan_id) = LOWER(s.plan_id)
    WHEN MATCHED THEN
    UPDATE SET
        pages = s.pages,
        source = s.source,
        sha256 = s.sha256,
        status = s.status,
        size_in_bytes = s.size_in_bytes,
        user_id = s.user_id,
        updated_at = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
    INSERT (
        plan_id,
        project_id,
        user_id,
        status,
        plan_name,
        plan_type,
        file_type,
        pages,
        size_in_bytes,
        source,
        sha256,
        created_at,
        updated_at
    )
    VALUES (
        s.plan_id,
        s.project_id,
        s.user_id,
        s.status,
        s.plan_name,
        s.plan_type,
        s.file_type,
        s.pages,
        s.size_in_bytes,
        s.source,
        s.sha256,
        CURRENT_TIMESTAMP(),
        CURRENT_TIMESTAMP()
    );
    """
    sha_256 = ''
    if plan_id:
        pdf_path = Path("/tmp/floor_plan.PDF")
        download_floorplan(plan_id, project_id, credentials, destination_path=pdf_path)
        sha_256 = sha256(pdf_path)
    if not plan_id:
        plan_id = payload_plan.plan_id
    plan_name, plan_type, file_type = '', '', ''
    if payload_plan:
        plan_name, plan_type, file_type = payload_plan.plan_name, payload_plan.plan_type, payload_plan.file_type
    if not n_pages:
        n_pages = 0
    if not GCS_URL_floorplan:
        GCS_URL_floorplan = ''
    if not size_in_bytes:
        size_in_bytes = 0
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("status", "STRING", status),
            bigquery.ScalarQueryParameter("plan_name", "STRING", plan_name),
            bigquery.ScalarQueryParameter("plan_type", "STRING", plan_type),
            bigquery.ScalarQueryParameter("file_type", "STRING", file_type),
            bigquery.ScalarQueryParameter("pages", "INT64", n_pages),
            bigquery.ScalarQueryParameter("source", "STRING", GCS_URL_floorplan),
            bigquery.ScalarQueryParameter("sha256", "STRING", sha_256),
            bigquery.ScalarQueryParameter("size_in_bytes", "INT64", size_in_bytes)
        ]
    )

    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output


def insert_project(payload_project, bigquery_client, credentials):
    GBQ_query = """
    MERGE `drywall_takeoff.projects` t
    USING (
        SELECT
            @project_id AS project_id,
            @project_name AS project_name,
            @project_location AS project_location,
            @FBM_branch AS FBM_branch,
            @project_type AS project_type,
            @project_area AS project_area,
            @contractor_name AS contractor_name,
            @created_by AS created_by
    ) s
    ON LOWER(t.project_id) = LOWER(s.project_id)
    WHEN NOT MATCHED THEN
    INSERT (
        project_id,
        project_name,
        project_location,
        FBM_branch,
        project_type,
        project_area,
        contractor_name,
        created_at,
        created_by
    )
    VALUES (
        s.project_id,
        s.project_name,
        s.project_location,
        s.FBM_branch,
        s.project_type,
        s.project_area,
        s.contractor_name,
        CURRENT_TIMESTAMP(),
        s.created_by
    );
    """
    job_config = dict(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_id", "STRING", payload_project.project_id),
            bigquery.ScalarQueryParameter("project_name", "STRING", payload_project.project_name),
            bigquery.ScalarQueryParameter("project_location", "STRING", payload_project.project_location),
            bigquery.ScalarQueryParameter("FBM_branch", "STRING", payload_project.FBM_branch),
            bigquery.ScalarQueryParameter("project_type", "STRING", payload_project.project_type),
            bigquery.ScalarQueryParameter("project_area", "STRING", payload_project.project_area),
            bigquery.ScalarQueryParameter("contractor_name", "STRING", payload_project.contractor_name),
            bigquery.ScalarQueryParameter("created_by", "STRING", payload_project.created_by)
        ]
    )

    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    GBQ_query = f"SELECT created_at FROM `{credentials["GBQServer"]["table_name_projects"]}` WHERE project_id = '{payload_project.project_id}'"
    query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
    created_at = list(query_output)[0].created_at.isoformat()
    return created_at


def floorplan_to_structured_2d(credentials, id_token, project_id, plan_id, user_id, page_number):
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }
    requests.post(
        f"{credentials["CloudRun"]["APIs"]["floorplan_to_structured_2d"]}/floorplan_to_structured_2d",
        headers=headers,
        json=dict(
            project_id=project_id,
            plan_id=plan_id,
            user_id=user_id,
            page_number=page_number
        )
    )


def load_UI_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.map(lambda x: float(x) if isinstance(x, Decimal) else x)
    df = df.map(lambda x: "null" if isinstance(x, int) and (pd.isna(x) or math.isnan(x) or math.isinf(x) or np.isnan(x) or np.isinf(x)) else x)
    df = df.map(lambda x: "null" if isinstance(x, float) and (pd.isna(x) or math.isnan(x) or math.isinf(x) or np.isnan(x) or np.isinf(x)) else x)
    df = df.map(lambda x: x.date().isoformat() if isinstance(x, datetime) else x)
    df = df.map(lambda x: x.isoformat() if isinstance(x, date) else x)
    df = df.map(lambda x: x.isoformat() if isinstance(x, time) else x)
    df = df.map(lambda x: b64encode(x).decode("utf-8") if isinstance(x, bytes) else x)

    return df


def enable_logging_on_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s"}',
        stream=sys.stdout
    )


def load_gcp_credentials() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("gcp.yaml", 'r') as f:
        credentials = yaml.load(f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials["service_drywall_account_key"]

    return credentials


def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("hyperparameters.yaml", 'r') as f:
        hyperparameters = yaml.load(f)

    return hyperparameters


app = FastAPI(title="Drywall Takeoff (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
bigquery_client = load_bigquery_client(CREDENTIALS)

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


@app.post("/generate_project")
async def generate_project(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    try:
        payload_project = PayloadProject(**parameters)
    except ValidationError:
        payload_project = PayloadProject(**body)
    created_at = insert_project(payload_project, bigquery_client, CREDENTIALS)
    logging.info(f"SYSTEM: New Project {payload_project.project_name} generated successfully")
    return respond_with_UI_payload(
        dict(
            project_id=payload_project.project_id,
            project_name=payload_project.project_name,
            created_at=created_at,
        )
    )


@app.post("/load_projects")
async def load_projects(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()

    GBQ_query = f"SELECT * FROM `{CREDENTIALS["GBQServer"]["table_name_projects"]}`"
    projects = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())

    logging.info("SYSTEM: Project Metaaata retrieved successfully")
    return respond_with_UI_payload(
        jsonable_encoder({
            "projects": projects
        })
    )


@app.post("/load_project_plans")
async def load_project_plans(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")

    query = f"""
        SELECT
            p.*,
            ARRAY(
                SELECT AS STRUCT *
                FROM `{CREDENTIALS["GBQServer"]["table_name_plans"]}` pl
                WHERE pl.project_id = p.project_id
            ) AS project_plans
        FROM `{CREDENTIALS["GBQServer"]["table_name_projects"]}` p
        WHERE LOWER(p.project_id) = LOWER(@project_id)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_id", "STRING", project_id)
        ]
    )

    query_job = bigquery_client.query(query, job_config=job_config)
    rows = list(query_job.result())

    if not rows:
        return respond_with_UI_payload(dict(project_metadata=dict(), project_plans=list()))

    row = rows[0]
    project_metadata = dict(row)
    project_plans = project_metadata.pop("project_plans", list())

    logging.info("SYSTEM: Project Plans Data retrieved successfully")
    return respond_with_UI_payload(
        jsonable_encoder({
            "project_metadata": project_metadata,
            "project_plans": project_plans
        })
    )


@app.post("/generate_floorplan_upload_signed_URL")
async def generate_floorplan_upload_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    payload_plan = parameters.get("plan") or body.get("plan")
    user_id = parameters.get("user_id") or body.get("user_id")
    payload_plan = PayloadPlan(**payload_plan)
    logging.info("SYSTEM: Received Signed Floorplan upload URL generation Request")

    insert_plan(
        project_id,
        user_id,
        "NOT STARTED",
        bigquery_client,
        CREDENTIALS,
        payload_plan=payload_plan
    )

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{payload_plan.plan_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="PUT",
        content_type="application/octet-stream",
    )

    return url


@app.post("/load_plan_pages")
async def load_plan_pages(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")

    GBQ_query = f"SELECT * FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
    query_output = bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).to_dataframe()
    dataframe = load_UI_dataframe(query_output)
    plan_metadata = dict()
    if dataframe.to_dict(orient="records"):
        plan_metadata = dataframe.to_dict(orient="records")[0]

    GBQ_query = f"SELECT * FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
    query_output = bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).to_dataframe()
    dataframe = load_UI_dataframe(query_output)
    plan_pages_data = dataframe.to_dict(orient="records")

    logging.info(f"SYSTEM: Plan Pages Data retrieved successfully")
    return respond_with_UI_payload(dict(plan_metadata=plan_metadata, plan_pages=plan_pages_data))


@app.post("/floorplan_to_2d")
async def floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    verbose = parameters.get("verbose") or body.get("verbose")
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")

    pdf_path = Path("/tmp/floor_plan.PDF")
    GCS_URL_floorplan = download_floorplan(plan_id, project_id, CREDENTIALS, destination_path=pdf_path)
    logging.info("SYSTEM: Floorplan Downloaded")
    plan_duplicate = is_duplicate(bigquery_client, CREDENTIALS, pdf_path, project_id)
    if plan_duplicate:
        delete_plan(CREDENTIALS, bigquery_client, plan_id, project_id)
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"tmp/{user_id.lower()}/{project_id.lower()}/{plan_id.lower()}/floorplan_structured_2d.json"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.delete()

    size_in_bytes = Path(pdf_path).stat().st_size
    floor_plan_paths_vector, floor_plan_paths_preprocessed = preprocess(pdf_path)
    insert_plan(
        project_id,
        user_id,
        "IN PROGRESS",
        bigquery_client,
        CREDENTIALS,
        plan_id=plan_id,
        size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
    )
    logging.info("SYSTEM: Floorplan Preprocessing Completed")

    walls_2d_all = dict(pages=list())
    status = "COMPLETED"
    vertex_ai_client, generation_config = load_vertex_ai_client(CREDENTIALS)
    vertex_ai_client_parameters = (vertex_ai_client, generation_config, CREDENTIALS["VertexAI"]["llm"]["max_retry"])
    try:
        id_token = load_floorplan_to_structured_2d_ID_token(CREDENTIALS)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = list()
            floorplan_baseline_page_sources = list()
            floorplan_page_sources = list()
            plan_types = list()
            for index, (floor_plan_vector, floor_plan_path) in enumerate(zip(floor_plan_paths_vector, floor_plan_paths_preprocessed)):
                plan_type = classify_plan(floor_plan_path, vertex_ai_client_parameters)
                plan_types.append(plan_type)
                floorplan_baseline_page_source = upload_floorplan(floor_plan_vector, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
                floorplan_baseline_page_sources.append(floorplan_baseline_page_source)
                floorplan_page_source = upload_floorplan(floor_plan_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
                floorplan_page_sources.append(floorplan_page_source)
                if plan_type["plan_type"].upper().find("FLOOR") == -1:
                    continue
                futures.append(
                    executor.submit(
                        floorplan_to_structured_2d,
                        CREDENTIALS,
                        id_token,
                        project_id,
                        plan_id,
                        user_id,
                        index
                    )
                )
            for page_number, (plan_type, _, floorplan_page_source) in enumerate(zip(plan_types, floorplan_baseline_page_sources, floorplan_page_sources)):
                if plan_type["plan_type"].upper().find("FLOOR") == -1:
                    continue
                timeout = from_unix_epoch() + 3600
                while from_unix_epoch() < timeout:
                    GBQ_query = f"SELECT scale, model_2d FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
                    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
                    if query_output:
                        break
                    sleep(2)
                walls_2d = json.loads(query_output[0].model_2d) if isinstance(query_output[0].model_2d, str) else query_output[0].model_2d
                if not walls_2d["polygons"] or not walls_2d["walls_2d"]:
                    GBQ_query = f"DELETE FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
                    bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()
                    continue
                GBQ_query = f"UPDATE `{CREDENTIALS["GBQServer"]["table_name_models"]}` SET source = '{floorplan_page_source}' WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
                bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()
                page = dict(
                    plan_id=plan_id,
                    page_number=page_number,
                    page_type=plan_type["plan_type"].upper(),
                    scale=query_output[0].scale,
                    walls_2d=walls_2d["walls_2d"],
                    polygons=walls_2d["polygons"],
                    **walls_2d["metadata"]
                )
                walls_2d_all["pages"].append(page)
    except Exception as e:
        logging.info(f"SYSTEM: Floorplan extraction failed with error: {e}")
        status = "FAILED"
    insert_plan(
        project_id,
        user_id,
        status,
        bigquery_client,
        CREDENTIALS,
        plan_id=plan_id,
        size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
    )

    with open("/tmp/floorplan_structured_2d.json", 'w') as f:
        json.dump(walls_2d_all, f, indent=4)
    blob.upload_from_filename("/tmp/floorplan_structured_2d.json")
    return respond_with_UI_payload(walls_2d_all)


@app.post("/load_2d_revision")
async def load_2d_revision(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    revision_number = parameters.get("revision_number") or body.get("revision_number")
    logging.info(f"SYSTEM: Received Floorplan 2D Model (Revision: {revision_number}) Load Request")

    GBQ_query = f"SELECT model FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_2d"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number} AND revision_number = {revision_number};"
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
    walls_2d_JSON = dict()
    if query_output and query_output[0].model is not None:
        walls_2d_JSON = json.loads(query_output[0].model)

    return respond_with_UI_payload(walls_2d_JSON)


@app.post("/load_available_revision_numbers_2d")
async def load_available_revision_numbers_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info(f"SYSTEM: Received Available Revisions Load Request for 2D Model")

    GBQ_query = f"SELECT revision_number FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_2d"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
    revision_numbers = list()
    if query_output:
        for revision in query_output:
            if revision.revision_number is not None:
                revision_numbers.append(revision.revision_number)

    return respond_with_UI_payload(revision_numbers)


@app.post("/load_2d_all")
async def load_2d_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number", '') or body.get("page_number", '')
    logging.info("SYSTEM: Received All Floorplan 2D Models Load Request")

    status = "IN PROGRESS"
    GBQ_query = f"SELECT pages FROM `{CREDENTIALS["GBQServer"]["table_name_plans"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
    if not list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()):
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())[0]
    n_pages = query_output.pages
    timeout = from_unix_epoch() + (n_pages * 120)
    while from_unix_epoch() < timeout:
        GBQ_query = f"SELECT status FROM `{CREDENTIALS["GBQServer"]["table_name_plans"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
        try:
            query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())[0]
            status = query_output.status
            if status == "COMPLETED":
                break
        except IndexError:
            return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
        sleep(2)
    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed within 15 minutes"), status_code=500)

    walls_2d_all = dict(pages=list())
    if page_number != '':
        query = f"""
            SELECT
                page_number,
                scale,
                model_2d
            FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}`
            WHERE
                LOWER(project_id) = LOWER(@project_id)
                AND LOWER(plan_id) = LOWER(@plan_id)
                AND page_number = @page_number
            ORDER BY page_number
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
                bigquery.ScalarQueryParameter("page_number", "INT64", page_number),
            ]
        )
    else:
        query = f"""
            SELECT
                page_number,
                scale,
                model_2d
            FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}`
            WHERE
                LOWER(project_id) = LOWER(@project_id)
                AND LOWER(plan_id) = LOWER(@plan_id)
            ORDER BY page_number
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
                bigquery.ScalarQueryParameter("plan_id", "STRING", plan_id),
            ]
        )
    query_job = bigquery_client.query(query, job_config=job_config)

    for row in query_job.result():
        if not row.model_2d:
            continue

        walls_2d = json.loads(row.model_2d) if isinstance(row.model_2d, str) else row.model_2d
        page = {
            "plan_id": plan_id,
            "page_number": row.page_number,
            "scale": row.scale,
            "walls_2d": walls_2d.get("walls_2d", list()),
            "polygons": walls_2d.get("polygons", list()),
            **walls_2d.get("metadata", dict()),
        }

        walls_2d_all["pages"].append(page)

    return respond_with_UI_payload(walls_2d_all)


@app.post("/update_floorplan_to_2d")
async def update_floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    scale = parameters.get("scale") or body.get("scale")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 2D Model Update Request")

    insert_model_2d(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, None, None, bigquery_client, CREDENTIALS)
    insert_model_2d_revision(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)
    logging.info("SYSTEM: Floorplan 2D Model Updated Successfully")


@app.post("/update_scale")
async def update_scale(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    scale = parameters.get("scale") or body.get("scale")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Scale Update Request")

    GBQ_query = f"UPDATE `{CREDENTIALS["GBQServer"]["table_name_models"]}` SET scale = '{scale}' WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
    bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()
    logging.info("SYSTEM: Scale Updated Successfully")


@app.post("/load_scale")
async def load_scale(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Scale Update Request")

    GBQ_query = f"SELECT scale FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())[0]
    return respond_with_UI_payload(dict(scale=query_output.scale))


@app.post("/floorplan_to_3d")
async def floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    scale = parameters.get("scale") or body.get("scale")
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 3D Model Generation Request")

    model_2d_path = "/tmp/walls_2d.json"
    with open(model_2d_path, 'w') as f:
        json.dump(walls_2d_JSON, f)
    polygons_path = "/tmp/polygons.json"
    with open(polygons_path, 'w') as f:
        json.dump(polygons_JSON, f)
    hyperparameters = load_hyperparameters()
    if not scale:
        GBQ_query = f"SELECT scale FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {index};"
        query_output = bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()
        scale = list(query_output)[0].scale
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    walls_3d, polygons_3d, walls_3d_path, polygons_3d_path = floor_plan_modeller_3d.extrapolate(scale, model_2d_path=model_2d_path, polygons_path=polygons_path)
    walls_3d, polygons_3d = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d, polygons_3d)
    gltf_paths = floor_plan_modeller_3d.gltf(model_2d_path=model_2d_path, polygons_path=polygons_path)
    model_3d_path = floor_plan_modeller_3d.save_plot_3d(walls_3d_path, polygons_3d_path)
    GBQ_query = f"SELECT model_2d.metadata FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {index};"
    query_output = bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()
    metadata = list(query_output)[0].metadata
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    upload_floorplan(model_3d_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
    for gltf_path in gltf_paths:
        upload_floorplan(gltf_path, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2), directory="gltf")
    insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)
    logging.info("SYSTEM: A 3D Model of the Floorplan Generated Successfully")

    return respond_with_UI_payload(dict(walls_3d=walls_3d, polygons=polygons_3d, metadata=metadata))


@app.post("/load_3d_revision")
async def load_3d_revision(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    revision_number = parameters.get("revision_number") or body.get("revision_number")
    logging.info(f"SYSTEM: Received Floorplan 3D Model (Revision: {revision_number}) Load Request")

    GBQ_query = f"SELECT model FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_3d"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND LOWER(user_id) = LOWER('{user_id}') AND page_number = {page_number} AND revision_number = {revision_number};"
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
    walls_3d_JSON = dict()
    if query_output and query_output[0].model is not None:
        walls_3d_JSON = json.loads(query_output[0].model)

    return respond_with_UI_payload(walls_3d_JSON)


@app.post("/load_available_revision_numbers_3d")
async def load_available_revision_numbers_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info(f"SYSTEM: Received Available Revisions Load Request for 3D Model")

    GBQ_query = f"SELECT revision_number FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_3d"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND LOWER(user_id) = LOWER('{user_id}') AND page_number = {page_number};"
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
    revision_numbers = list()
    if query_output:
        for revision in query_output:
            if revision.revision_number is not None:
                revision_numbers.append(revision.revision_number)

    return respond_with_UI_payload(revision_numbers)


@app.post("/load_3d_all")
async def load_3d_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received All Floorplan 3D Models Load Request")

    walls_3d_all = dict(pages=list())
    GBQ_query = f"SELECT page_number, scale, model_3d FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
    query_output = bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).to_dataframe()
    dataframe = load_UI_dataframe(query_output)
    for page_number, scale, model_3d in zip(dataframe["page_number"], dataframe["scale"], dataframe["model_3d"]):
        GBQ_query = f"SELECT model_2d.metadata FROM `drywall_takeoff.models` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
        query_output = bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()
        metadata = list(query_output)[0].metadata
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        walls_3d = json.loads(model_3d) if isinstance(model_3d, str) else model_3d
        page = dict(
            plan_id=plan_id,
            page_number=page_number,
            walls_3d=walls_3d["walls_3d"],
            polygons=walls_3d["polygons"],
            scale=scale,
            **metadata,
        )
        walls_3d_all["pages"].append(page)

    return respond_with_UI_payload(walls_3d_all)


@app.post("/update_floorplan_to_3d")
async def update_floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d = parameters.get("walls_3d") or body.get("walls_3d")
    polygons_3d = parameters.get("polygons") or body.get("polygons")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    scale = parameters.get("scale") or body.get("scale")
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 3D Model Update Request")

    insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)
    insert_model_3d_revision(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)
    logging.info("SYSTEM: Floorplan 3D Model Updated Successfully")


@app.post("/generate_drywall_overlaid_floorplan_download_signed_URL")
async def generate_drywall_overlaid_floorplan_download_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    index = parameters.get("page_number") or body.get("page_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received Signed Floorplan download URL generation Request")

    status = "IN PROGRESS"
    GBQ_query = f"SELECT pages FROM `{CREDENTIALS["GBQServer"]["table_name_plans"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
    if not list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()):
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())[0]
    n_pages = query_output.pages
    timeout = from_unix_epoch() + (n_pages * 120)
    while from_unix_epoch() < timeout:
        GBQ_query = f"SELECT status FROM `{CREDENTIALS["GBQServer"]["table_name_plans"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
        try:
            query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())[0]
            status = query_output.status
            if status == "COMPLETED":
                break
        except IndexError:
            return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
        sleep(2)
    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed within 15 minutes"), status_code=500)

    GBQ_query = f"SELECT target_drywalls FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {index};"
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
    drywall_overlaid_floorplan_source_path = list(query_output)[0].target_drywalls
    _, _, _, blob_path = drywall_overlaid_floorplan_source_path.split('/', 3)

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="GET",
    )

    return url


@app.post("/remove_floorplan")
async def remove_floorplan(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received a Floorplan Deletion Request")

    delete_floorplan(project_id, plan_id, user_id, bigquery_client, CREDENTIALS)
    logging.info("SYSTEM: Floorplan Deleted Successfully")


@app.post("/compute_takeoff")
async def compute_takeoff(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d_JSON = parameters.get("walls_3d", list()) or body.get("walls_3d", list())
    polygons_JSON = parameters.get("polygons", list()) or body.get("polygons", list())
    index = parameters.get("page_number") or body.get("page_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    revision_number = parameters.get("revision_number", '') or body.get("revision_number", '')
    logging.info("SYSTEM: Received a Drywall Takeoff computation Request")

    GBQ_query = f"SELECT scale FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {index};"
    query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())[0]
    scale = query_output.scale
    pdf_path = Path("/tmp/floor_plan.PDF")
    download_floorplan(plan_id, project_id, CREDENTIALS, destination_path=pdf_path)

    if not walls_3d_JSON:
        if revision_number:
            GBQ_query = f"SELECT model FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_3d"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {index} AND revision_number = {revision_number};"
            walls_3d_JSON = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())[0].model
        else:
            GBQ_query = f"SELECT model_3d FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {index};"
            walls_3d_JSON = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())[0].model_3d
    
        if walls_3d_JSON is None:
            walls_3d_JSON = list()

    hyperparameters = load_hyperparameters()
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    if scale != "1/4``=1`0``":
        pixel_aspect_ratio_new = floor_plan_modeller_3d.compute_pixel_aspect_ratio(scale, hyperparameters["pixel_aspect_ratio_to_feet"])
        walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.recompute_dimensions_walls_and_polygons(walls_3d_JSON, polygons_JSON, pixel_aspect_ratio_new, pdf_path)
    walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d_JSON, polygons_JSON)
    drywall_takeoff = dict(total=dict(roof=0, wall=0), per_drywall=dict(roof=defaultdict(lambda: 0), wall=defaultdict(lambda: 0)))
    for wall in walls_3d_JSON:
        surface_area = wall["height"] * wall["length"]
        drywall_count = 0
        for drywall in wall["surfaces_drywall"]:
            if drywall["enabled"]:
                waste_factor = drywall["waste_factor"]
                if isinstance(waste_factor, float):
                    waste_factor = float(waste_factor)
                elif waste_factor.find('%') != -1:
                    if waste_factor.find('-') != -1:
                        waste_factor = float(waste_factor.strip('%').split('-')[1]) / 100
                    else:
                        waste_factor = float(waste_factor.strip('%')) / 100
                else:
                    try:
                        waste_factor = float(waste_factor)
                    except ValueError:
                        waste_factor = 0
                drywall_takeoff["per_drywall"]["wall"][drywall["type"]] += surface_area * (1 + waste_factor)
                drywall_count += 1
        drywall_takeoff["total"]["wall"] += drywall_count * surface_area
    for polygon in polygons_JSON:
        surface_area = floor_plan_modeller_3d.compute_updated_area_polygon(
            polygon["vertices"],
            polygon["area"],
            polygon["slope"],
            polygon["tilt_axis"]
        )
        waste_factor = polygon["surface_drywall"]["waste_factor"]
        if isinstance(waste_factor, float):
            waste_factor = float(waste_factor)
        elif waste_factor.find('%') != -1:
            if waste_factor.find('-') != -1:
                waste_factor = float(waste_factor.strip('%').split('-')[1]) / 100
            else:
                waste_factor = float(waste_factor.strip('%')) / 100
        else:
            try:
                waste_factor = float(waste_factor)
            except ValueError:
                waste_factor = 0
        drywall_takeoff["per_drywall"]["roof"][polygon["surface_drywall"]["type"]] += surface_area * (1 + waste_factor)
        drywall_takeoff["total"]["roof"] += surface_area

    drywall_takeoff["total"]["wall"] = round(drywall_takeoff["total"]["wall"], 2)
    drywall_takeoff["total"]["roof"] = round(drywall_takeoff["total"]["roof"], 2)
    for key in drywall_takeoff["per_drywall"]["wall"]:
        drywall_takeoff["per_drywall"]["wall"][key] = round(drywall_takeoff["per_drywall"]["wall"][key], 2)
    for key in drywall_takeoff["per_drywall"]["roof"]:
        drywall_takeoff["per_drywall"]["roof"][key] = round(drywall_takeoff["per_drywall"]["roof"][key], 2)

    insert_takeoff(drywall_takeoff, index, plan_id, user_id, project_id, revision_number, bigquery_client, CREDENTIALS)
    logging.info("SYSTEM: Drywall Takeoff Computed Successfully for the provided Floorplan")
    return respond_with_UI_payload(drywall_takeoff)


@app.get("/insert_templates")
async def insert_templates():
    def parse_fire_rating(description: str):
        if "TYPE C" in description:
            return "Type C"
        if "TYPE X" in description:
            return "Type X"
        return None

    def parse_lightweight(description: str):
        return "LITE" in description.upper()

    def parse_wide_stretch(description: str):
        return "WIDE-STRETCH" in description.upper()

    def parse_thickness(description: str):
        match = re.search(r'(\d+\/\d+)"', description)
        if match:
            fraction = match.group(1)
            numerator, denominator = fraction.split("/")
            return float(numerator) / float(denominator)
        return None

    def generate_random_colors(n, seed=0):
        rng = np.random.default_rng(seed)
        total_colors = 256**3
        excluded_index = 255 * 256 * 256 + 0 * 256 + 0

        indices = rng.choice(
            total_colors - 1,
            size=n,
            replace=False
        )

        indices = np.where(indices >= excluded_index, indices + 1, indices)
        colors = list()
        for index in indices:
            r = index // (256 * 256)
            g = (index // 256) % 256
            b = index % 256
            colors.append((int(r), int(g), int(b)))

        return [dict(r=int(color[0]), g=int(color[1]), b=int(color[2])) for color in colors]

    dataframe = pd.read_excel("Drywall_P_Code_20260122.xlsx")
    rows_to_insert = list()
    product_color_codes = generate_random_colors(dataframe.size)

    for (_, row), product_color_code in zip(dataframe.iterrows(), product_color_codes):
        if pd.isna(row["user10"]) or pd.isna(row["user11"]) or pd.isna(row["PRODUCT_CAT_CODE"]) or pd.isna(row["PRODUCT_CAT_DESC"]) or not isinstance(row["PRODUCT_CAT_CODE"], int):
            continue
        sku_description = str(row["user11"]).upper()

        parsed_row = {
            "sku_id": row["user10"],
            "sku_description": row["user11"],
            "product_cat_code": int(row["PRODUCT_CAT_CODE"]),
            "product_cat_description": row["PRODUCT_CAT_DESC"],
            "thickness_inches": parse_thickness(sku_description),
            "fire_rating": parse_fire_rating(sku_description),
            "is_lightweight": parse_lightweight(sku_description),
            "is_wide_stretch": parse_wide_stretch(sku_description),
            "color_code": product_color_code
        }

        rows_to_insert.append(parsed_row)

    error = bigquery_client.insert_rows_json(CREDENTIALS["GBQServer"]["table_name_sku"], rows_to_insert)
    if error:
        logging.info(f"SYSTEM: Template insertion failed with error: {error}")
    else:
        logging.info("SYSTEM: Templates successfully inserted")