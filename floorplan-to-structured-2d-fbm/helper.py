import logging
import json
import sys
import os
from pathlib import Path
from ruamel.yaml import YAML
from time import sleep

from random import uniform

import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud.storage import Client as CloudStorageClient
from google.cloud import bigquery
from fastapi.encoders import jsonable_encoder
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, DeadlineExceeded

from transcriber import Transcriber


def load_vertex_ai_client(credentials, region="us-central1"):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = GenerativeModel(credentials["VertexAI"]["llm"]["model_name"])
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config

def transcribe(credentials, hyperparameters, floor_plan_path):
    transcriber = Transcriber(credentials, hyperparameters)
    return transcriber.transcribe(floor_plan_path, [0, 1, -1, -2])

def upload_floorplan(plan_path, plan_id, project_id, credentials, index=None, directory=None):
    client = CloudStorageClient()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if directory:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{directory}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{directory}/{blob_object_name}"
    else:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{blob_object_name}"
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(plan_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"

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

def download_floorplan(user_id, plan_id, project_id, credentials, index, destination_path="/tmp/floor_plan_wall_processed.png"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/floor_plan.png"
    blob = bucket.blob(blob_path)

    destination_path = Path(destination_path)
    destination_path = destination_path.parent.joinpath(project_id).joinpath(plan_id).joinpath(user_id).joinpath(destination_path.name)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_path)
    return destination_path

def load_bigquery_client(credentials):
    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    return bigquery_client

def bigquery_run(credentials, bigquery_client, GBQ_query, job_config=dict()):
    job_config = bigquery.QueryJobConfig(
        destination_encryption_configuration=bigquery.EncryptionConfiguration(
            kms_key_name=credentials["GBQServer"]["KMS_key"]
        ),
        **job_config
    )
    query_output = bigquery_client.query(GBQ_query, job_config=job_config)
    return query_output

def insert_model_2d(
    model_2d,
    scale,
    page_number,
    plan_id,
    user_id,
    project_id,
    target_drywalls,
    bigquery_client,
    credentials
    ):
    GBQ_query = """
    MERGE `drywall_takeoff.models` t
    USING (
        SELECT
            @plan_id AS plan_id,
            @project_id AS project_id,
            @user_id AS user_id,
            @page_number AS page_number,
            @model_2d AS model_2d,
            @scale AS scale,
            @target_drywalls AS target_drywalls,
    ) s
    ON LOWER(t.project_id) = LOWER(s.project_id) AND LOWER(t.plan_id) = LOWER(s.plan_id) AND t.page_number = s.page_number
    WHEN MATCHED THEN
    UPDATE SET
        model_2d = s.model_2d,
        scale = COALESCE(NULLIF(s.scale, ''), t.scale),
        user_id = @user_id,
        updated_at = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
    INSERT (
        plan_id,
        project_id,
        user_id,
        page_number,
        scale,
        model_2d,
        model_3d,
        takeoff,
        target_drywalls,
        created_at,
        updated_at
    )
    VALUES (
        s.plan_id,
        s.project_id,
        s.user_id,
        s.page_number,
        s.scale,
        s.model_2d,
        JSON '{}',
        JSON '{}',
        s.target_drywalls,
        CURRENT_TIMESTAMP(),
        CURRENT_TIMESTAMP()
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
            bigquery.ScalarQueryParameter("target_drywalls", "STRING", target_drywalls),
        ]
    )

    query_output = bigquery_run(credentials, bigquery_client, GBQ_query, job_config=job_config).result()
    return query_output

def load_templates(bigquery_client, credentials):
    GBQ_query = f"SELECT * FROM `{credentials["GBQServer"]["table_name_sku"]}`"
    product_templates = list(bigquery_run(credentials, bigquery_client, GBQ_query).result())

    logging.info("SYSTEM: Product Templates retrieved successfully")
    product_templates_target = list()
    cached_templates_sku = list()
    for product_template in product_templates:
        product_template = dict(product_template)
        if product_template["sku_id"] in cached_templates_sku:
            continue
        cached_templates_sku.append(product_template["sku_id"])
        product_template["sku_variant"] = f"{product_template["sku_id"]} - {product_template["sku_description"]}"
        product_template["color_code"] = [product_template["color_code"]['b'], product_template["color_code"]['g'], product_template["color_code"]['r']]
        product_templates_target.append(product_template)
    return jsonable_encoder(product_templates_target)

def phoenix_call(generate_content_lambda, max_retry=5, base_delay=1.0, pydantic_model=None):
    n_iterations = 0
    temperature = 0
    while n_iterations < max_retry:
        try:
            response = generate_content_lambda(temperature)
            if pydantic_model:
                json_response = json.loads(response.text.strip("`json").replace("{{", '{').replace("}}", '}'))
                response_json_pydantic = pydantic_model(**json_response)
                return response_json_pydantic, json_response
            return response.text
        except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded) as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            sleep_time = base_delay * (2 ** (n_iterations - 1)) + uniform(0, 0.5)
            sleep(sleep_time)
            logging.warning(f"SYSTEM: {e}: RETRYING ...")
        except Exception as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            temperature = min(0.5 * (n_iterations + 1) / max_retry, 0.5)
            logging.warning(f"SYSTEM: Response Generation/Parsing failed with ERROR: {e}")
            logging.warning(f"SYSTEM: RETRYING with TEMPERATURE: {temperature}")