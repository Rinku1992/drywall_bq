import logging
import json
import sys
import os
import requests
from pathlib import Path
from ruamel.yaml import YAML
from time import sleep
import datetime

from random import uniform
from PIL import Image
import numpy as np
import math

import geoip2.database as geoip2_database
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud.storage import Client as CloudStorageClient
from google.cloud import bigquery
from fastapi.encoders import jsonable_encoder
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, DeadlineExceeded
from vertexai.generative_models import Content, Part
from vertexai.caching import CachedContent

from transcriber import Transcriber
from prompt import FEEDBACK_GENERATOR


def load_vertex_ai_client(credentials, ip_address, prompts=None, default_region="us-central1"):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    region = load_nearest_region(
        ip_address,
        credentials["geolite_database"],
        credentials["VertexAI"]["llm"]["available_regions"],
        default_region=default_region
    )
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = lambda system_instruction: GenerativeModel(
        credentials["VertexAI"]["llm"]["model_name"],
        system_instruction=system_instruction
    )
    is_cached = False
    if prompts and GenerativeModel(credentials["VertexAI"]["llm"]["model_name"]).count_tokens(prompts).total_tokens >= 1024:
        is_cached = True
        cached_content = CachedContent.create(
            model_name=credentials["VertexAI"]["llm"]["model_name"],
            contents=prompts,
            ttl=datetime.timedelta(minutes=60),
            display_name="drywall_predictor_cache"
        )
        vertex_ai_client = GenerativeModel.from_cached_content(cached_content)
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config, is_cached

def load_nearest_region(ip_address, geolite_database, available_regions, default_region="us-central1"):
    def _compute_haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2):
        R = 6371
        d_latitude = math.radians(latitude_2-latitude_1)
        d_longitude = math.radians(longitude_2-longitude_1)
        a = math.sin(d_latitude/2)**2 + math.cos(math.radians(latitude_1)) * math.cos(math.radians(latitude_2)) * math.sin(d_longitude/2)**2
        return 2*R*math.asin(math.sqrt(a))

    if not ip_address or "," not in ip_address:
        return default_region
    if ip_address and "," in ip_address:
        ip_address = ip_address.split(",")[0].strip()
    geoip2_reader = geoip2_database.Reader(geolite_database)
    try:
        response = geoip2_reader.city(ip_address)
        (response.location.latitude, response.location.longitude, response.country.iso_code)
        nearest_region = None
        minimum_distance = float("inf")
        for region, (latitude, longitude) in available_regions.items():
            distance = _compute_haversine_distance(response.location.latitude, response.location.longitude, latitude, longitude)
            if distance < minimum_distance:
                minimum_distance = distance
                nearest_region = region
        return nearest_region
    except Exception:
        return default_region

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
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
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
    page_section_number,
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
            @page_section_number AS page_section_number,
            @model_2d AS model_2d,
            @scale AS scale,
            @target_drywalls AS target_drywalls,
    ) s
    ON LOWER(t.project_id) = LOWER(s.project_id) AND LOWER(t.plan_id) = LOWER(s.plan_id) AND t.page_number = s.page_number and t.page_section_number = s.page_section_number
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
        page_section_number,
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
        s.page_section_number,
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
            bigquery.ScalarQueryParameter("page_section_number", "STRING", page_section_number),
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

def phoenix_call(generate_content_lambda, max_retry=5, base_delay=1.0, pydantic_model=None, verify_field_counts=None):
    n_iterations = 0
    temperature = 0
    exceptions = list()
    feedback_prompt = ''
    while n_iterations < max_retry:
        try:
            response = generate_content_lambda(feedback_prompt, temperature)
            if pydantic_model:
                json_response = json.loads(response.text.strip("`json").replace("{{", '{').replace("}}", '}'))
                if verify_field_counts:
                    for field, count in verify_field_counts.items():
                        if len(json_response[field]) != count:
                            raise ValueError(f"Predicted {field} count: {len(json_response[field])} does not match with the expected number: {count}")
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
            exceptions.append(e)
            system_feedback = [Part.from_text(FEEDBACK_GENERATOR.format(max_retry=max_retry, exceptions=exceptions))]
            feedback_prompt = Content(role="model", parts=system_feedback)
            temperature = min(0.5 * (n_iterations + 1) / max_retry, 0.5)
            logging.warning(f"SYSTEM: Response Generation/Parsing failed with ERROR: {e}")
            logging.warning(f"SYSTEM: RETRYING with TEMPERATURE: {temperature}")

def load_section_from_page(wall_segmented_path, floor_plan_path, bounding_box_offset, section_name):
    offset_top_left_X, offset_top_left_Y = bounding_box_offset["offset_top_left"]
    offset_bottom_right_X, offset_bottom_right_Y = bounding_box_offset["offset_bottom_right"]
    offset_top_left_X = max(offset_top_left_X - 0.05, 0)
    offset_top_left_Y = max(offset_top_left_Y - 0.05, 0)
    offset_bottom_right_X  = min(offset_bottom_right_X + 0.05, 1)
    offset_bottom_right_Y  = min(offset_bottom_right_Y + 0.05, 1)
    canvas = Image.open(wall_segmented_path)
    canvas = canvas.convert("RGB")
    width_in_pixels, height_in_pixels = canvas.size
    canvas_original = Image.open(floor_plan_path)
    canvas_original = canvas_original.convert("RGB")
    width_in_pixels_original, height_in_pixels_original = canvas_original.size

    canvas = canvas.resize((width_in_pixels_original, height_in_pixels_original), Image.Resampling.NEAREST)
    image = np.array(canvas).copy()
    LEFT = round(offset_top_left_X * width_in_pixels_original)
    TOP = round(offset_top_left_Y * height_in_pixels_original)
    BOTTOM = round(offset_bottom_right_Y * height_in_pixels_original)
    RIGHT = round(offset_bottom_right_X * width_in_pixels_original)
    image[:TOP, :] = 255
    image[:, :LEFT] = 255
    image[:, RIGHT:] = 255
    image[BOTTOM:, :] = 255
    canvas = Image.fromarray(image)
    canvas = canvas.resize((width_in_pixels, height_in_pixels), Image.Resampling.NEAREST)
    wall_segmented_path_sectioned = wall_segmented_path.parent.joinpath(f"{wall_segmented_path.stem}_sectioned_{section_name}").with_suffix(".png")
    canvas.save(wall_segmented_path_sectioned, format="png")

    return str(wall_segmented_path_sectioned)

def polygon_to_structured_2d(credentials, query_json):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["polygon_to_structured_2d"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token

    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{credentials["CloudRun"]["APIs"]["polygon_to_structured_2d"]}/polygon_to_structured_2d",
        headers=headers,
        json=query_json
    )
    return response.status_code, response.content