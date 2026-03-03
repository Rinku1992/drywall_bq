from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import sleep
import logging

from random import uniform

from google.cloud import vision
from google.oauth2 import service_account
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, DeadlineExceeded
import cv2

__all__ = ["Transcriber"]


class Transcriber:

    def __init__(self, credentials, hyperparameters):
        self._credentials = credentials
        self._hyperparameters = hyperparameters
        self._transcription_block_centroids = dict()
        self._transcription_block_tiles = dict()

    def _image_to_string(
        self,
        image_to_string_client,
        v_stride_index,
        h_stride_index,
        kernel_parameters,
        n_horizontal_strides,
        image_array,
        output_path,
        max_retry
    ):
        X1 = h_stride_index * kernel_parameters["stride"]
        X2 = X1 + kernel_parameters["width"]
        Y1 = v_stride_index * kernel_parameters["stride"]
        Y2 = Y1 + kernel_parameters["height"]

        cropped = image_array[Y1:Y2, X1:X2]
        cv2.imwrite(f"/tmp/{output_path}_{str((v_stride_index*n_horizontal_strides)+h_stride_index).zfill(3)}.png", cropped)

        with open(f"/tmp/{output_path}_{str((v_stride_index*n_horizontal_strides)+h_stride_index).zfill(3)}.png", "rb") as f:
            content = f.read()
        image = vision.Image(content=content)

        n_iterations = 0
        base_delay = 1.0
        while n_iterations < max_retry:
            try:
                response = image_to_string_client.document_text_detection(image=image)
                break
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

        response_json = json.loads(response.__class__.to_json(response))
        if response_json["textAnnotations"]:
            text = response_json["textAnnotations"][0]["description"]
            bounding_box_A, bounding_box_B, bounding_box_C, bounding_box_D  = response_json["textAnnotations"][0]["boundingPoly"]["vertices"]
            centroid_x = (bounding_box_A['x'] + bounding_box_B['x'] + bounding_box_C['x'] + bounding_box_D['x']) / 4
            centroid_y = (bounding_box_A['y'] + bounding_box_B['y'] + bounding_box_C['y'] + bounding_box_D['y']) / 4
            self._transcription_block_centroids[text] = [centroid_x + X1, centroid_y + Y1]
            self._transcription_block_tiles[text] = [v_stride_index, h_stride_index]

        with open(f"/tmp/{output_path}_{str((v_stride_index*n_horizontal_strides)+h_stride_index).zfill(3)}.json", "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)

    @property
    def transciption_block_tiles(self):
        return self._transcription_block_tiles

    def transciption_block_tiles_row_major(self, row_indexes):
        n_rows = max([index_row_major[0] for index_row_major in self._transcription_block_tiles.values()]) + 1
        transcription_block_tiles_row_major = dict()
        for row_index in row_indexes:
            if row_index < 0:
                row_index = n_rows - row_index
            transcription_block_tiles_row_major[row_index + 1] = list()
            for text, index_row_major in self._transcription_block_tiles.items():
                if row_index == index_row_major[0]:
                    transcription_block_tiles_row_major[row_index + 1].append(text)

        return transcription_block_tiles_row_major

    def transcribe(self, image_path: Path, filter_transciption_block_tiles_row_major_indexes=None, max_retry=5):
        credentials = service_account.Credentials.from_service_account_file(self._credentials["service_drywall_account_key"])
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)

        kernel_parameters = self._hyperparameters["modelling"]["kernel"]
        image = cv2.imread(image_path)
        n_horizontal_strides = (image.shape[1] // kernel_parameters["stride"]) + 1
        n_vertical_strides = (image.shape[0] // kernel_parameters["stride"]) + 1

        futures = list()
        with ThreadPoolExecutor(max_workers=1) as executor:
            for v_stride_index in range(n_vertical_strides):
                for h_stride_index in range(n_horizontal_strides):
                    futures.append(executor.submit(
                        self._image_to_string,
                        vision_client,
                        v_stride_index,
                        h_stride_index,
                        kernel_parameters,
                        n_horizontal_strides,
                        image,
                        "ocr_clip",
                        max_retry,
                    ))

            [future.result() for future in futures]
        if filter_transciption_block_tiles_row_major_indexes:
            return self._transcription_block_centroids, self.transciption_block_tiles_row_major(filter_transciption_block_tiles_row_major_indexes)
        return self._transcription_block_centroids