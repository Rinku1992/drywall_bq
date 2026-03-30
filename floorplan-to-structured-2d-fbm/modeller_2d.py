from copy import deepcopy
import cv2
import numpy as np
np.random.seed(0)
import math
import json
import logging
import xml.etree.ElementTree as ET
from base64 import b64encode
from PIL import Image
from pathlib import Path
from collections import defaultdict
import subprocess
from concurrent.futures import ThreadPoolExecutor

from fractions import Fraction
from skimage.morphology import skeletonize
from vertexai.generative_models import Part, Content
from shapely.geometry import Polygon

from floor_plan import FloorPlan
from prompt import (
    DRYWALL_PREDICTOR_CALIFORNIA,
    SCALE_AND_CEILING_HEIGHT_DETECTOR,
    WALL_RECTIFIER,
    SHAPE_RECTIFIER,
    CEILING_CHOICES,
    DrywallPredictorCaliforniaResponse,
    ScaleAndCeilingHeightDetectorResponse,
    WallRectifierResponse,
    ShapeRectifierResponse,
)
from helper import (
    load_vertex_ai_client,
    phoenix_call,
    polygon_to_structured_2d,
    extract_wall_dimension_candidates,
    log_json,
)

__all__ = ["FloorPlan2D"]


class FloorPlan2D(FloorPlan):

    def __init__(self, credentials, hyperparameters, drywall_templates):
        super().__init__(hyperparameters)

        self._credentials = credentials
        self._hyperparameters = hyperparameters
        self._drywall_templates = drywall_templates
        self._width_in_feet = self._hyperparameters["modelling"]["width_in_feet"]
        self._height_in_feet = self._hyperparameters["modelling"]["height_in_feet"]
        self._scale = self._hyperparameters["modelling"]["scale"]
        self._walls_2d = list()
        self._polygons = list()

    def reload(self):
        self._walls_2d = list()
        self._polygons = list()

    @classmethod
    def load_vertex_ai_clients(cls, credentials, client_ip_address, drywall_templates):
        is_cached = dict()
        vertex_ai_client_drywall_prediction, generation_config, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[DRYWALL_PREDICTOR_CALIFORNIA.format(drywall_templates=drywall_templates)]
        )
        is_cached["DRYWALL_PREDICTOR_CALIFORNIA"] = cache_enabled
        vertex_ai_client_metadata_extraction, _, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[SCALE_AND_CEILING_HEIGHT_DETECTOR]
        )
        is_cached["SCALE_AND_CEILING_HEIGHT_DETECTOR"] = cache_enabled
        vertex_ai_client_wall_rectification, _, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[WALL_RECTIFIER]
        )
        is_cached["WALL_RECTIFIER"] = cache_enabled
        vertex_ai_client_shape_rectification, _, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[SHAPE_RECTIFIER]
        )
        is_cached["SHAPE_RECTIFIER"] = cache_enabled
        vertex_ai_clients = (
            vertex_ai_client_drywall_prediction,
            vertex_ai_client_metadata_extraction,
            vertex_ai_client_wall_rectification,
            vertex_ai_client_shape_rectification
        )
        return is_cached, vertex_ai_clients, generation_config

    def from_vertex_ai_clients(self, is_cached, vertex_ai_clients, generation_config):
        self._vertex_ai_generation_config = generation_config
        self._is_cached = is_cached
        self._vertex_ai_client_drywall_prediction = vertex_ai_clients[0]
        self._vertex_ai_client_metadata_extraction = vertex_ai_clients[1]
        self._vertex_ai_client_wall_rectification = vertex_ai_clients[2]
        self._vertex_ai_client_shape_rectification = vertex_ai_clients[3]

    def _close_jagged_openings(
        self,
        wall_lines,
        tolerance_distance=5,
        n_steps=2500,
    ):
        def load_wall_line_index_random(n_wall_lines, wall_line_indices_prior):
            while True:
                wall_line_index = np.random.randint(n_wall_lines)
                if wall_line_index in wall_line_indices_prior:
                    continue
                break
            return wall_line_index

        if not wall_lines:
            return wall_lines
        try:
            wall_lines = wall_lines.tolist()
        except:
            ...

        wall_line_indices_prior = list()
        for n_step in range(len(wall_lines)):
            wall_lines_new = deepcopy(wall_lines)
            if n_step >= len(wall_lines):
                wall_line_indices_prior = list()
            wall_line_index = load_wall_line_index_random(len(wall_lines), wall_line_indices_prior)
            wall_line_indices_prior.append(wall_line_index)
            reference_line = wall_lines[wall_line_index]
            X1, Y1, X2, Y2 = reference_line[0]
            reference_line_type = self.classify_line(X1, Y1, X2, Y2)
            if reference_line_type not in ["vertical", "horizontal"]:
                continue
            for target_line in wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_line[0]
                target_line_type = self.classify_line(target_X1, target_Y1, target_X2, target_Y2)
                if target_line_type not in ["vertical", "horizontal"]:
                    continue
                if reference_line_type == target_line_type:
                    euclidean_distance_1 = np.hypot(target_X1 - X2, target_Y1 - Y2)
                    euclidean_distance_2 = np.hypot(target_X2 - X1, target_Y2 - Y1)
                    if min(euclidean_distance_1, euclidean_distance_2) <= tolerance_distance:
                        if euclidean_distance_1 <= euclidean_distance_2:
                            new_line_horizontal = [[X2, Y2, target_X1, Y2]]
                            new_line_vertical = [[target_X1, Y2, target_X1, target_Y1]]
                            wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            break
                        if euclidean_distance_2 <= euclidean_distance_1:
                            new_line_horizontal = [[X1, Y1, target_X2, Y1]]
                            new_line_vertical = [[target_X2, Y1, target_X2, target_Y2]]
                            wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            break
                else:
                    if reference_line_type == "horizontal":
                        chebyshev_distance_1 = max(abs(target_X1 - X1), min(abs(target_Y1 - Y1), abs(target_Y2 - Y1)))
                        chebyshev_distance_2 = max(abs(target_X1 - X2), min(abs(target_Y1 - Y2), abs(target_Y2 - Y2)))
                        if min(chebyshev_distance_1, chebyshev_distance_2) <= tolerance_distance:
                            if chebyshev_distance_1 <= chebyshev_distance_2:
                                if Y1 >= min(target_Y1, target_Y2) and Y1 <= max(target_Y1, target_Y2):
                                    new_line = [[X1, Y1, target_X1, Y1]]
                                    wall_lines_new.append(new_line)
                                else:
                                    new_line_horizontal = [[X1, Y1, target_X1, Y1]]
                                    new_line_vertical = [[target_X1, Y1, target_X1, target_Y1 if abs(target_Y1 - Y1) < abs(target_Y2 - Y1) else target_Y2]]
                                    wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            else:
                                if Y2 >= min(target_Y1, target_Y2) and Y2 <= max(target_Y1, target_Y2):
                                    new_line = [[X2, Y2, target_X1, Y2]]
                                    wall_lines_new.append(new_line)
                                else:
                                    new_line_horizontal = [[X2, Y2, target_X1, Y2]]
                                    new_line_vertical = [[target_X1, Y2, target_X1, target_Y1 if abs(target_Y1 - Y2) < abs(target_Y2 - Y2) else target_Y2]]
                                    wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            break
                    else:
                        chebyshev_distance_1 = max(abs(target_Y1 - Y1), min(abs(target_X1 - X1), abs(target_X2 - X1)))
                        chebyshev_distance_2 = max(abs(target_Y1 - Y2), min(abs(target_X1 - X2), abs(target_X2 - X2)))
                        if min(chebyshev_distance_1, chebyshev_distance_2) <= tolerance_distance:
                            if chebyshev_distance_1 <= chebyshev_distance_2:
                                if X1 >= min(target_X1, target_X2) and X1 <= max(target_X1, target_X2):
                                    new_line = [[X1, Y1, X1, target_Y1]]
                                    wall_lines_new.append(new_line)
                                else:
                                    new_line_vertical = [[X1, Y1, X1, target_Y1]]
                                    new_line_horizontal = [[X1, target_Y1, target_X1 if abs(target_X1 - X1) < abs(target_X2 - X1) else target_X2, target_Y1]]
                                    wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            else:
                                if X2 >= min(target_X1, target_X2) and X2 <= max(target_X1, target_X2):
                                    new_line = [[X2, Y2, X2, target_Y1]]
                                    wall_lines_new.append(new_line)
                                else:
                                    new_line_vertical = [[X2, Y2, X2, target_Y1]]
                                    new_line_horizontal = [[X2, target_Y1, target_X1 if abs(target_X1 - X1) < abs(target_X2 - X1) else target_X2, target_Y1]]
                                    wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            break
            wall_lines = wall_lines_new

        return wall_lines

    def _sniff_and_split_orthogonal(
        self,
        wall_lines,
        tolerance_distance=5,
    ):
        if not wall_lines:
            return wall_lines
        try:
            wall_lines = wall_lines.tolist()
        except:
            ...

        horizontal_wall_lines = list()
        vertical_wall_lines = list()
        wall_lines_splitted = list()
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line_type = self.classify_line(X1, Y1, X2, Y2)
            if wall_line_type == "horizontal":
                horizontal_wall_lines.append(wall_line)
            elif wall_line_type == "vertical":
                vertical_wall_lines.append(wall_line)
            else:
                wall_lines_splitted.append(wall_line)

        for horizontal_wall_line in horizontal_wall_lines:
            X1, Y1, X2, Y2 = horizontal_wall_line[0]
            split_found = False
            for target_line in vertical_wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_line[0]
                if target_X1 > min(X1, X2) and target_X1 < max(X1, X2):
                    if min(target_Y1, target_Y2) - tolerance_distance <= Y1 <= max(target_Y1, target_Y2) + tolerance_distance:
                        wall_lines_splitted.append([[X1, Y1, target_X1, Y1]])
                        wall_lines_splitted.append([[target_X1, Y1, X2, Y2]])
                        split_found = True
                        break
            if not split_found:
                wall_lines_splitted.append(horizontal_wall_line)

        for vertical_wall_line in vertical_wall_lines:
            X1, Y1, X2, Y2 = vertical_wall_line[0]
            split_found = False
            for target_line in horizontal_wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_line[0]
                if target_Y1 > min(Y1, Y2) and target_Y1 < max(Y1, Y2):
                    if min(target_X1, target_X2) - tolerance_distance <= X1 <= max(target_X1, target_X2) + tolerance_distance:
                        wall_lines_splitted.append([[X1, Y1, X1, target_Y1]])
                        wall_lines_splitted.append([[X1, target_Y1, X2, Y2]])
                        split_found  = True
                        break
            if not split_found:
                wall_lines_splitted.append(vertical_wall_line)

        return wall_lines_splitted

    def _close_wall_openings_deterministic(
        self,
        wall_lines,
        tolerance_distance=25
    ):
        try:
            wall_lines = wall_lines.tolist()
        except:
            ...
    
        new_lines = set()

        for i, reference_line in enumerate(wall_lines):
            X1 = min(reference_line[0][0], reference_line[0][2])
            X2 = max(reference_line[0][0], reference_line[0][2])
            Y1 = min(reference_line[0][1], reference_line[0][3])
            Y2 = max(reference_line[0][1], reference_line[0][3])

            reference_line_type = self.classify_line(X1, Y1, X2, Y2)
            if reference_line_type not in ["vertical", "horizontal"]:
                continue

            for j, target_line in enumerate(wall_lines):
                if i == j:
                    continue

                t_X1 = min(target_line[0][0], target_line[0][2])
                t_X2 = max(target_line[0][0], target_line[0][2])
                t_Y1 = min(target_line[0][1], target_line[0][3])
                t_Y2 = max(target_line[0][1], target_line[0][3])

                target_line_type = self.classify_line(t_X1, t_Y1, t_X2, t_Y2)
                if target_line_type not in ["vertical", "horizontal"]:
                    continue

                if reference_line_type == target_line_type == "horizontal":
                    if abs(t_Y1 - Y1) <= tolerance_distance:
                        if abs(t_X1 - X1) <= tolerance_distance:
                            new_lines.add((X1, Y1, X1, t_Y1))
                        if abs(t_X2 - X2) <= tolerance_distance:
                            new_lines.add((X2, Y2, X2, t_Y1))

                elif reference_line_type == target_line_type == "vertical":
                    if abs(t_X1 - X1) <= tolerance_distance:
                        if abs(t_Y1 - Y1) <= tolerance_distance:
                            new_lines.add((X1, Y1, t_X1, Y1))
                        if abs(t_Y2 - Y2) <= tolerance_distance:
                            new_lines.add((X2, Y2, t_X1, Y2))

        for x1, y1, x2, y2 in new_lines:
            wall_lines.append([[x1, y1, x2, y2]])

        return wall_lines

    def _group_lines(self, wall_lines, tolerance_distance=5):
        """Cluster lines into groups of nearby, same-orientation lines."""
        n = len(wall_lines)
        adjacency = defaultdict(list)

        for i in range(n):
            x1, y1, x2, y2 = wall_lines[i][0]
            type_i = self.classify_line(x1, y1, x2, y2)
            if not type_i:
                continue
            for j in range(i + 1, n):
                x3, y3, x4, y4 = wall_lines[j][0]
                type_j = self.classify_line(x3, y3, x4, y4)
                if type_i != type_j:
                    continue
                dists = [
                    np.hypot(x1 - x3, y1 - y3),
                    np.hypot(x1 - x4, y1 - y4),
                    np.hypot(x2 - x3, y2 - y3),
                    np.hypot(x2 - x4, y2 - y4),
                ]
                if min(dists) <= tolerance_distance:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        visited = set()
        clusters = list()
        for i in range(n):
            if i not in visited:
                stack = [i]
                cluster = list()
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        cluster.append(node)
                        stack.extend(adjacency[node])
                clusters.append(cluster)

        return clusters

    def _merge_cluster(self, cluster, wall_lines):
        """Merge a cluster of lines into one smooth line."""
        points = list()
        for idx in cluster:
            x1, y1, x2, y2 = wall_lines[idx][0]
            points.extend([(x1, y1), (x2, y2)])

        x1, y1, x2, y2 = wall_lines[cluster[0]][0]
        line_type = self.classify_line(x1, y1, x2, y2)

        if line_type == "horizontal":
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            y_mean = int(round(np.mean(ys)))
            return [[min(xs), y_mean, max(xs), y_mean]]

        elif line_type == "vertical":
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_mean = int(round(np.mean(xs)))
            return [[x_mean, min(ys), x_mean, max(ys)]]

        elif line_type == "inclined":
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            m, c = np.polyfit(xs, ys, 1)
            x_min, x_max = xs.min(), xs.max()
            return [[int(x_min), int(m*x_min + c), int(x_max), int(m*x_max + c)]]

        return None

    def _jagged_to_smooth_lines_deterministic(
        self,
        wall_lines,
        tolerance_distance=15,
    ):
        """Convert jagged wall lines into smoother merged lines deterministically."""
        try:
            wall_lines = wall_lines.tolist()
        except:
            wall_lines = list(wall_lines)

        clusters = self._group_lines(wall_lines, tolerance_distance)
        merged_lines = list()

        for cluster in clusters:
            merged = self._merge_cluster(cluster, wall_lines)
            if merged:
                merged_lines.append(merged)

        return merged_lines

    def _remove_orthogonal_overlap(self, reference_line, target_line, reference_line_type="horizontal", tolerance_distance=5):
        x1_reference, y1_reference, x2_reference, y2_reference = reference_line[0]
        x1_target, y1_target, x2_target, y2_target = target_line[0]
        if reference_line_type == "horizontal":
            x_target = int(np.median([x1_target, x2_target]))
            y_reference = int(np.median([y1_reference, y2_reference]))
            if (x_target >= x1_reference and x_target <= x2_reference) and (y_reference >= y1_target and y_reference <= y2_target):
                if abs(y_reference - y1_target) < abs(y_reference - y2_target) and abs(y_reference - y1_target) <= tolerance_distance:
                    return [[x1_target, y_reference, x2_target, y2_target]]
                if abs(y_reference - y2_target) < abs(y_reference - y1_target) and abs(y_reference - y2_target) <= tolerance_distance:
                    return [[x1_target, y1_target, x2_target, y_reference]]
                return [[x1_target, y1_target, x2_target, y2_target]]
            return [[x1_target, y1_target, x2_target, y2_target]]
        if reference_line_type == "vertical":
            y_target = int(np.median([y1_target, y2_target]))
            x_reference = int(np.median([x1_reference, x2_reference]))
            if (y_target >= y1_reference and y_target <= y2_reference) and (x_reference >= x1_target and x_reference <= x2_target):
                if abs(x_reference - x1_target) < abs(x_reference - x2_target) and abs(x_reference - x1_target) <= tolerance_distance:
                    return [[x_reference, y1_target, x2_target, y2_target]]
                if abs(x_reference - x2_target) < abs(x_reference - x1_target) and abs(x_reference - x2_target) <= tolerance_distance:
                    return [[x1_target, y1_target, x_reference, y1_target]]
                return [[x1_target, y1_target, x2_target, y2_target]]
            return [[x1_target, y1_target, x2_target, y2_target]]

    def _no_orthogonal_overlap(
        self,
        wall_lines,
        tolerance_disance=100,
        n_steps=5000
    ):
        def load_wall_line_index_random(n_wall_lines, wall_line_indices_prior):
            while True:
                wall_line_index = np.random.randint(n_wall_lines)
                if wall_line_index in wall_line_indices_prior:
                    continue
                break
            return wall_line_index

        if wall_lines is None:
            return wall_lines
        wall_line_indices_prior = list()
        for n_step in range(len(wall_lines)):
            wall_lines_new = deepcopy(wall_lines)
            if n_step >= len(wall_lines):
                wall_line_indices_prior = list()
            wall_line_index = load_wall_line_index_random(len(wall_lines), wall_line_indices_prior)
            wall_line_indices_prior.append(wall_line_index)
            reference_wall_line = wall_lines[wall_line_index]
            x1_reference, y1_reference, x2_reference, y2_reference = reference_wall_line[0]
            reference_line_type = self.classify_line(x1_reference, y1_reference, x2_reference, y2_reference)
            if reference_line_type not in ["vertical", "horizontal"]:
                continue
            for target_wall_line in wall_lines:
                x1_target, y1_target, x2_target, y2_target = target_wall_line[0]
                target_line_type = self.classify_line(x1_target, y1_target, x2_target, y2_target)
                if target_line_type not in ["vertical", "horizontal"]:
                    continue
                if reference_line_type == "horizontal" and target_line_type == "vertical":
                    target_wall_line_new = self._remove_orthogonal_overlap(reference_wall_line, target_wall_line, reference_line_type="horizontal", tolerance_distance=tolerance_disance)
                    wall_lines_new.remove(target_wall_line)
                    wall_lines_new.append(target_wall_line_new)
                if reference_line_type == "vertical" and target_line_type == "horizontal":
                    target_wall_line_new = self._remove_orthogonal_overlap(reference_wall_line, target_wall_line, reference_line_type="vertical", tolerance_distance=tolerance_disance)
                    wall_lines_new.remove(target_wall_line)
                    wall_lines_new.append(target_wall_line_new)
            wall_lines = wall_lines_new

        return wall_lines

    def _thin_edges(self, binary_image):
        _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

        skeleton = np.zeros(binary.shape, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        img = binary.copy()

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()

            done = (cv2.countNonZero(img) == 0)

        return skeleton

    def _load_topology(self, floor_plan_wall_segmented_binary: np.ndarray):
        def _binary_to_bool(img: np.ndarray) -> np.ndarray:
            return img > 0

        def _bool_to_binary(img_bool: np.ndarray) -> np.ndarray:
            return ((img_bool == False).astype(np.uint8) * 255)

        floor_plan_wall_segmented_bool = _binary_to_bool(floor_plan_wall_segmented_binary)
        floor_plan_topology_bool = skeletonize(floor_plan_wall_segmented_bool)
        floor_plan_topology_binary = _bool_to_binary(floor_plan_topology_bool)

        return floor_plan_topology_binary

    def _preprocessing(self, image_BGR, floor_plan_path, scale, max_split=5):
        _, thresh = cv2.threshold(image_BGR, 50, 255, cv2.THRESH_BINARY_INV)

        edges_thinned = self._thin_edges(thresh)
        edges = cv2.Canny(edges_thinned, 50, 100, apertureSize=3)

        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        floor_plan_topology_binary = self._load_topology(edges)
        lines = self.detect_lines(edges)
        lines = self.normalize(lines)
        if lines is not None:
            lines = self._jagged_to_smooth_lines_deterministic(lines)
            lines = self._close_jagged_openings(lines)
            lines = self._close_wall_openings_deterministic(lines)
            for _ in range(3):
                lines = self._topology_guided_extend_and_conquer(lines, floor_plan_topology_binary)
                lines = self._topology_guided_closure_open_lines(lines, floor_plan_topology_binary)
            for _ in range(max_split):
                lines = self._sniff_and_split_orthogonal(lines)
                lines = self._deduplicate_lines(lines)
            lines = self._remove_invalid(lines)

            lines = self._topology_guided_closure_open_lines_dead_end(lines, maximum_length=150)
            lines = self._sniff_and_split_orthogonal(lines)
            lines = self._deduplicate_lines(lines)
            lines = self._remove_invalid(lines)
            shapes = self.disconnected_shapes(lines)

            is_valid_futures = list()
            with ThreadPoolExecutor(max_workers=4) as executor:
                for shape in shapes:
                    is_valid_futures.append(executor.submit(
                        self._is_shape_valid,
                        shape,
                        scale,
                        floor_plan_path,
                    ))
                is_shape_valid = [future.result() for future in is_valid_futures]
            lines = list()
            for shape, is_valid in zip(shapes, is_shape_valid):
                if is_valid and len(shape) > 4:
                    lines.extend(self._merge_nearest_neighbor(shape))

        return lines

    def _is_shape_valid(self, lines, scale, floor_plan_path):
        scale_x, scale_y = scale
        wall_lines_structured = list()
        canvas = cv2.imread(floor_plan_path)
        for line in lines:
            X1, Y1, X2, Y2 = line[0]
            polygons_drywall = self._extrude_polygon_perimeter(line, scale, outer_drywall_surface="INVALID")
            for drywall in polygons_drywall:
                pts = np.array([
                    [drywall["coordinates"][0]['x'], drywall["coordinates"][0]['y']],
                    [drywall["coordinates"][1]['x'], drywall["coordinates"][1]['y']],
                    [drywall["coordinates"][2]['x'], drywall["coordinates"][2]['y']],
                    [drywall["coordinates"][3]['x'], drywall["coordinates"][3]['y']]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                canvas_to_overlay = canvas.copy()
                cv2.fillPoly(canvas_to_overlay, pts=[pts], color=(0, 0, 255))
                canvas = cv2.addWeighted(canvas_to_overlay, 0.7, canvas, 0.3, 0)
            cv2.line(canvas, (round(scale_x * X1), round(scale_y * Y1)), (round(scale_x * X2), round(scale_y * Y2)), (0, 0, 255), 2)
            wall_lines_structured.append(dict(wall=[{'x': round(scale_x * X1), 'y': round(scale_y * Y1)}, {'x': round(scale_x * X2), 'y': round(scale_y * Y2)}]))
        _, canvas_buffer_array = cv2.imencode(".png", canvas)
        bytes_canvas = canvas_buffer_array.tobytes()
        query = Content(role="user", parts=[
            Part.from_text(json.dumps(wall_lines_structured)),
            Part.from_data(data=bytes_canvas, mime_type="image/png")
        ])
        try:
            if self._is_cached["SHAPE_RECTIFIER"]:
                _, is_valid = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_shape_rectification.generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=ShapeRectifierResponse,
                )
            else:
                _, is_valid = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_shape_rectification(SHAPE_RECTIFIER).generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=ShapeRectifierResponse,
                )
            if is_valid["confidence"] > 0.9:
                return is_valid["is_valid"]
            return True
        except Exception as e:
            logging.warning(f"SYSTEM: Wall validator failed with error: {e}")
            return True

    def _merge_nearest_neighbor(self, wall_lines, tolerance=500):
        wall_lines_closed_dead_end = deepcopy(wall_lines)
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            open_ends = self.is_open(wall_line, wall_lines)
            if 'A' in open_ends:
                nearest_neighbor = self.nearest_neighbor(wall_line, 'A', wall_lines, tolerance=tolerance)
                if nearest_neighbor:
                    X1_nearest, Y1_nearest, X2_nearest, Y2_nearest = nearest_neighbor[0]
                    X1_new, Y1_new = X1, Y1
                    if math.hypot(X1_new - X1_nearest, Y1_new - Y1_nearest) < math.hypot(X1_new - X2_nearest, Y1_new - Y2_nearest):
                        wall_lines_closed_dead_end.append([[X1_new, Y1_new, X1_nearest, Y1_nearest]])
                    else:
                        wall_lines_closed_dead_end.append([[X1_new, Y1_new, X2_nearest, Y2_nearest]])
            if 'B' in open_ends:
                nearest_neighbor = self.nearest_neighbor(wall_line, 'B', wall_lines, tolerance=tolerance)
                if nearest_neighbor:
                    X1_nearest, Y1_nearest, X2_nearest, Y2_nearest = nearest_neighbor[0]
                    X1_new, Y1_new = X2, Y2
                    if math.hypot(X1_new - X1_nearest, Y1_new - Y1_nearest) < math.hypot(X1_new - X2_nearest, Y1_new - Y2_nearest):
                        wall_lines_closed_dead_end.append([[X1_new, Y1_new, X1_nearest, Y1_nearest]])
                    else:
                        wall_lines_closed_dead_end.append([[X1_new, Y1_new, X2_nearest, Y2_nearest]])

        return wall_lines_closed_dead_end

    def _topology_guided_closure_open_lines_dead_end(self, wall_lines, maximum_length=1000, tolerance=5):
        wall_lines_closed_dead_end = list()
        canvas = np.ones((1080, 1920), dtype=np.uint8) * 255
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            cv2.line(canvas, (X1, Y1), (X2, Y2), 0, 1)
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            open_ends = self.is_open(wall_line, wall_lines)
            if open_ends:
                orientation = self.classify_line(X1, Y1, X2, Y2)
                if orientation == "horizontal":
                    Y = int(np.median([Y1, Y2]))
                    if 'A' in open_ends:
                        target_X1 = X1 - np.argmin(canvas[Y - tolerance: Y + tolerance, : X1].mean(axis=0)[::-1])
                        pixel_value = canvas[Y - tolerance: Y + tolerance, : X1].mean(axis=0)[::-1][np.argmin(canvas[Y - tolerance: Y + tolerance, : X1].mean(axis=0)[::-1])]
                        if target_X1 > 0 and pixel_value != 255 and abs(X1 - target_X1) <= maximum_length:
                            wall_lines_closed_dead_end.append([[target_X1, Y, X1, Y]])
                    if 'B' in open_ends:
                        target_X2 = X2 + np.argmin(canvas[Y - tolerance: Y + tolerance, X2+1:].mean(axis=0))
                        pixel_value = canvas[Y - tolerance: Y + tolerance, X2+1:].mean(axis=0)[np.argmin(canvas[Y - tolerance: Y + tolerance, X2+1:].mean(axis=0))]
                        if target_X2 < 1920 and pixel_value != 255 and abs(target_X2 - X2) <= maximum_length:
                            wall_lines_closed_dead_end.append([[X2, Y, target_X2, Y]])
                if orientation == "vertical":
                    X = int(np.median([X1, X2]))
                    if 'A' in open_ends:
                        target_Y1 = Y1 - np.argmin(canvas[: Y1, X - tolerance: X + tolerance].mean(axis=1)[::-1])
                        pixel_value = canvas[: Y1, X - tolerance: X + tolerance].mean(axis=1)[::-1][np.argmin(canvas[: Y1, X - tolerance: X + tolerance].mean(axis=1)[::-1])]
                        if target_Y1 > 0  and pixel_value != 255 and abs(Y1 - target_Y1) <= maximum_length:
                            wall_lines_closed_dead_end.append([[X, target_Y1, X, Y1]])
                    if 'B' in open_ends:
                        target_Y2 = Y2 + np.argmin(canvas[Y2+1:, X - tolerance: X + tolerance].mean(axis=1))
                        pixel_value = canvas[Y2+1:, X - tolerance: X + tolerance].mean(axis=1)[np.argmin(canvas[Y2+1:, X - tolerance: X + tolerance].mean(axis=1))]
                        if target_Y2 < 1080  and pixel_value != 255 and abs(target_Y2 - Y2) <= maximum_length:
                            wall_lines_closed_dead_end.append([[X, Y2, X, target_Y2]])
                wall_lines_closed_dead_end.append(wall_line)
            else:
                wall_lines_closed_dead_end.append(wall_line)

        return wall_lines_closed_dead_end

    def _topology_guided_extend_and_conquer(self, lines, floor_plan_topology_binary, extension_maximum=10, tolerance=2):
        extended_lines = list()
        for line in lines:
            X1, Y1, X2, Y2 = line[0]
            orientation = self.classify_line(X1, Y1, X2, Y2)
            if orientation == "horizontal":
                Y = int(np.median([Y1, Y2]))
                for n_pixels in range(extension_maximum):
                    X1_new = max(0, X1 - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y - tolerance: Y + tolerance, X1_new]==0):
                        X1 = X1_new
                for n_pixels in range(extension_maximum):
                    X2_new = min(1920 - 1, X2 + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y - tolerance: Y + tolerance, X2_new]==0):
                        X2 = X2_new
                extended_lines.append([[X1, Y1, X2, Y2]])
            elif orientation == "vertical":
                X = int(np.median([X1, X2]))
                for n_pixels in range(extension_maximum):
                    Y1_new = max(0, Y1 - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y1_new, X - tolerance: X + tolerance]==0):
                        Y1 = Y1_new
                for n_pixels in range(extension_maximum):
                    Y2_new = min(1080 - 1, Y2 + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y2_new, X - tolerance: X + tolerance]==0):
                        Y2 = Y2_new
                extended_lines.append([[X1, Y1, X2, Y2]])
            else:
                extended_lines.append(line)

        return extended_lines

    def _topology_guided_closure_open_lines(self, wall_lines, floor_plan_topology_binary, extension_maximum=20, tolerance=10):
        if not wall_lines:
            return wall_lines
        try:
            wall_lines = wall_lines.tolist()
        except:
            wall_lines = list(wall_lines)

        def extend_open_line(wall_line, wall_line_type, open_end_type):
            extended_lines = list()
            X1, Y1, X2, Y2 = wall_line[0]
            if wall_line_type == "horizontal" and open_end_type == 'A':
                Y = int(np.median([Y1, Y2]))
                target_X1, target_Y1, target_X2, target_Y2 = X1, np.inf, X1, np.inf
                for n_pixels in range(extension_maximum):
                    Y_new = max(0, Y - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y_new, X1 - tolerance: X1 + tolerance]==0):
                        if target_Y2 == np.inf:
                            target_Y2 = Y_new
                        else:
                            target_Y1 = Y_new
                if target_Y1 != np.inf and target_Y2 != np.inf and math.hypot(0, target_Y1 - target_Y2) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                target_X1, target_Y1, target_X2, target_Y2 = X1, np.inf, X1, np.inf
                for n_pixels in range(extension_maximum):
                    Y_new = min(1080 - 1, Y + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y_new, X1 - tolerance: X1 + tolerance]==0):
                        if target_Y1 == np.inf:
                            target_Y1 = Y_new
                        else:
                            target_Y2 = Y_new
                if target_Y1 != np.inf and target_Y2 != np.inf and math.hypot(0, target_Y1 - target_Y2) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])

            if wall_line_type == "horizontal" and open_end_type == 'B':
                Y = int(np.median([Y1, Y2]))
                target_X1, target_Y1, target_X2, target_Y2 = X2, np.inf, X2, np.inf
                for n_pixels in range(extension_maximum):
                    Y_new = max(0, Y - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y_new, X2 - tolerance: X2 + tolerance]==0):
                        if target_Y2 == np.inf:
                            target_Y2 = Y_new
                        else:
                            target_Y1 = Y_new
                if target_Y1 != np.inf and target_Y2 != np.inf and math.hypot(0, target_Y1 - target_Y2) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                target_X1, target_Y1, target_X2, target_Y2 = X2, np.inf, X2, np.inf
                for n_pixels in range(extension_maximum):
                    Y_new = min(1080 - 1, Y + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y_new, X2 - tolerance: X2 + tolerance]==0):
                        if target_Y1 == np.inf:
                            target_Y1 = Y_new
                        else:
                            target_Y2 = Y_new
                if target_Y1 != np.inf and target_Y2 != np.inf and math.hypot(0, target_Y1 - target_Y2) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])

            if wall_line_type == "vertical" and open_end_type == 'A':
                X = int(np.median([X1, X2]))
                target_X1, target_Y1, target_X2, target_Y2 = np.inf, Y1, np.inf, Y1
                for n_pixels in range(extension_maximum):
                    X_new = max(0, X - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y1 - tolerance: Y1 + tolerance, X_new]==0):
                        if target_X2 == np.inf:
                            target_X2 = X_new
                        else:
                            target_X1 = X_new
                if target_X1 != np.inf and target_X2 != np.inf and math.hypot(target_X1 - target_X2, 0) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                target_X1, target_Y1, target_X2, target_Y2 = np.inf, Y1, np.inf, Y1
                for n_pixels in range(extension_maximum):
                    X_new = min(1920 - 1, X + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y1 - tolerance: Y1 + tolerance, X_new]==0):
                        if target_X1 == np.inf:
                            target_X1 = X_new
                        else:
                            target_X2 = X_new
                if target_X1 != np.inf and target_X2 != np.inf and math.hypot(target_X1 - target_X2, 0) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])

            if wall_line_type == "vertical" and open_end_type == 'B':
                X = int(np.median([X1, X2]))
                target_X1, target_Y1, target_X2, target_Y2 = np.inf, Y2, np.inf, Y2
                for n_pixels in range(extension_maximum):
                    X_new = max(0, X - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y2 - tolerance: Y2 + tolerance, X_new]==0):
                        if target_X2 == np.inf:
                            target_X2 = X_new
                        else:
                            target_X1 = X_new
                if target_X1 != np.inf and target_X2 != np.inf and math.hypot(target_X1 - target_X2, 0) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                target_X1, target_Y1, target_X2, target_Y2 = np.inf, Y2, np.inf, Y2
                for n_pixels in range(extension_maximum):
                    X_new = min(1920 - 1, X + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y2 - tolerance: Y2 + tolerance, X_new]==0):
                        if target_X1 == np.inf:
                            target_X1 = X_new
                        else:
                            target_X2 = X_new
                if target_X1 != np.inf and target_X2 != np.inf and math.hypot(target_X1 - target_X2, 0) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])

            return extended_lines

        horizontal_wall_lines = list()
        vertical_wall_lines = list()
        wall_lines_closed = list()

        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            line_type = self.classify_line(X1, Y1, X2, Y2)

            if line_type == "horizontal":
                horizontal_wall_lines.append(wall_line)
            elif line_type == "vertical":
                vertical_wall_lines.append(wall_line)
            else:
                wall_lines_closed.append(wall_line)

        for horizontal_wall_line in horizontal_wall_lines:
            open_ends = self.is_open(horizontal_wall_line, wall_lines, tolerance=tolerance)
            if open_ends and 'A' in open_ends:
                extended_lines = extend_open_line(horizontal_wall_line, "horizontal", 'A')
                if extended_lines:
                    wall_lines_closed.extend(extended_lines)
            if open_ends and 'B' in open_ends:
                extended_lines = extend_open_line(horizontal_wall_line, "horizontal", 'B')
                if extended_lines:
                    wall_lines_closed.extend(extended_lines)
            wall_lines_closed.append(horizontal_wall_line)
        for vertical_wall_line in vertical_wall_lines:
            open_ends = self.is_open(vertical_wall_line, wall_lines, tolerance=tolerance)
            if open_ends and 'A' in open_ends:
                extended_lines = extend_open_line(vertical_wall_line, "vertical", 'A')
                if extended_lines:
                    wall_lines_closed.extend(extended_lines)
            if open_ends and 'B' in open_ends:
                extended_lines = extend_open_line(vertical_wall_line, "vertical", 'B')
                if extended_lines:
                    wall_lines_closed.extend(extended_lines)
            wall_lines_closed.append(vertical_wall_line)

        return wall_lines_closed

    def _remove_invalid(self, wall_lines, open_tolerance_threshold=2, length_tolerance_threshold=10):
        valid_wall_lines = list()
        canvas = np.ones((1080, 1920), dtype=np.uint8) * 255
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            cv2.line(canvas, (X1, Y1), (X2, Y2), 0, 1)
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            if math.hypot(X1 - X2, Y1 - Y2) <= length_tolerance_threshold:
                continue
            open_ends = self.is_open(wall_line, wall_lines, tolerance=open_tolerance_threshold)
            if open_ends and 'A' in open_ends and 'B' in open_ends:
                orientation = self.classify_line(X1, Y1, X2, Y2)
                if orientation == "horizontal":
                    is_extended = False
                    pixel_distance_UP_A = np.argmin(canvas[: Y1, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1)[::-1])
                    pixel_value_UP_A = canvas[: Y1, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1)[::-1][np.argmin(canvas[: Y1, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1)[::-1])]
                    pixel_distance_UP_A = pixel_distance_UP_A if pixel_value_UP_A != 255 else np.inf
                    pixel_distance_DOWN_A = np.argmin(canvas[Y1 + 1:, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1))
                    pixel_value_DOWN_A = canvas[Y1 + 1:, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1)[np.argmin(canvas[Y1 + 1:, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1))]
                    pixel_distance_DOWN_A = pixel_distance_DOWN_A if pixel_value_DOWN_A != 255 else np.inf
                    pixel_distance_LEFT_A = np.argmin(canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, : X1].mean(axis=0)[::-1])
                    pixel_value_LEFT_A = canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, : X1].mean(axis=0)[::-1][np.argmin(canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, : X1].mean(axis=0)[::-1])]
                    pixel_distance_LEFT_A = pixel_distance_LEFT_A if pixel_value_LEFT_A != 255 else np.inf
                    if min(pixel_distance_UP_A, pixel_distance_DOWN_A, pixel_distance_LEFT_A) <= 500:
                        if np.argmin([pixel_distance_UP_A, pixel_distance_DOWN_A, pixel_distance_LEFT_A]) == 0 and pixel_value_UP_A != 255:
                            valid_wall_lines.append([[X1, Y1 - pixel_distance_UP_A, X1, Y1]])
                            is_extended = True
                        if np.argmin([pixel_distance_UP_A, pixel_distance_DOWN_A, pixel_distance_LEFT_A]) == 1 and pixel_value_DOWN_A != 255:
                            valid_wall_lines.append([[X1, Y1, X1, Y1 + pixel_distance_DOWN_A]])
                            is_extended = True
                        if np.argmin([pixel_distance_UP_A, pixel_distance_DOWN_A, pixel_distance_LEFT_A]) == 2 and pixel_value_LEFT_A != 255:
                            valid_wall_lines.append([[X1 - pixel_distance_LEFT_A, Y1, X1, Y1]])
                            is_extended = True
                    pixel_distance_UP_B = np.argmin(canvas[: Y2, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1)[::-1])
                    pixel_value_UP_B = canvas[: Y2, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1)[::-1][np.argmin(canvas[: Y2, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1)[::-1])]
                    pixel_distance_UP_B = pixel_distance_UP_B if pixel_value_UP_B != 255 else np.inf
                    pixel_distance_DOWN_B = np.argmin(canvas[Y2 + 1:, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1))
                    pixel_value_DOWN_B = canvas[Y2 + 1:, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1)[np.argmin(canvas[Y2 + 1:, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1))]
                    pixel_distance_DOWN_B = pixel_distance_DOWN_B if pixel_value_DOWN_B != 255 else np.inf
                    pixel_distance_RIGHT_B = np.argmin(canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, X2 + 1:].mean(axis=0)[::-1])
                    pixel_value_RIGHT_B = canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, X2 + 1:].mean(axis=0)[::-1][np.argmin(canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, X2 + 1:].mean(axis=0)[::-1])]
                    pixel_distance_RIGHT_B = pixel_distance_RIGHT_B if pixel_value_RIGHT_B != 255 else np.inf
                    if min(pixel_distance_UP_B, pixel_distance_DOWN_B, pixel_distance_RIGHT_B) <= 500:
                        if np.argmin([pixel_distance_UP_B, pixel_distance_DOWN_B, pixel_distance_RIGHT_B]) == 0 and pixel_value_UP_B != 255:
                            valid_wall_lines.append([[X2, Y2 - pixel_distance_UP_B, X2, Y2]])
                            is_extended = True
                        if np.argmin([pixel_distance_UP_B, pixel_distance_DOWN_B, pixel_distance_RIGHT_B]) == 1 and pixel_value_DOWN_B != 255:
                            valid_wall_lines.append([[X2, Y2, X2, Y2 + pixel_distance_DOWN_B]])
                            is_extended = True
                        if np.argmin([pixel_distance_UP_B, pixel_distance_DOWN_B, pixel_distance_RIGHT_B]) == 2 and pixel_value_RIGHT_B != 255:
                            valid_wall_lines.append([[X2, Y2, X2 + pixel_distance_RIGHT_B, Y2]])
                            is_extended = True
                    if is_extended:
                        valid_wall_lines.append(wall_line)
                if orientation == "vertical":
                    is_extended = False
                    pixel_distance_LEFT_A = np.argmin(canvas[Y1 - open_tolerance_threshold: Y1 + open_tolerance_threshold, :X1].mean(axis=0)[::-1])
                    pixel_value_LEFT_A = canvas[Y1 - open_tolerance_threshold: Y1 + open_tolerance_threshold, :X1].mean(axis=0)[::-1][np.argmin(canvas[Y1 - open_tolerance_threshold: Y1 + open_tolerance_threshold, :X1].mean(axis=0)[::-1])]
                    pixel_distance_LEFT_A = pixel_distance_LEFT_A if pixel_value_LEFT_A != 255 else np.inf
                    pixel_distance_RIGHT_A = np.argmin(canvas[Y1 - open_tolerance_threshold:Y1 + open_tolerance_threshold, X1 + 1:].mean(axis=0))
                    pixel_value_RIGHT_A = canvas[Y1 - open_tolerance_threshold:Y1 + open_tolerance_threshold, X1 + 1:].mean(axis=0)[np.argmin(canvas[Y1 - open_tolerance_threshold:Y1 + open_tolerance_threshold, X1 + 1:].mean(axis=0))]
                    pixel_distance_RIGHT_A = pixel_distance_RIGHT_A if pixel_value_RIGHT_A != 255 else np.inf
                    pixel_distance_UP_A = np.argmin(canvas[:Y1, X1 - open_tolerance_threshold:X1 + open_tolerance_threshold].mean(axis=1)[::-1])
                    pixel_value_UP_A = canvas[:Y1, X1 - open_tolerance_threshold:X1 + open_tolerance_threshold].mean(axis=1)[::-1][np.argmin(canvas[:Y1, X1 - open_tolerance_threshold:X1 + open_tolerance_threshold].mean(axis=1)[::-1])]
                    pixel_distance_UP_A = pixel_distance_UP_A if pixel_value_UP_A != 255 else np.inf
                    if min(pixel_distance_LEFT_A, pixel_distance_RIGHT_A, pixel_distance_UP_A) <= 500:
                        if np.argmin([pixel_distance_LEFT_A, pixel_distance_RIGHT_A, pixel_distance_UP_A]) == 0 and pixel_value_LEFT_A != 255:
                            valid_wall_lines.append([[X1 - pixel_distance_LEFT_A, Y1, X1, Y1]])
                            is_extended = True
                        if np.argmin([pixel_distance_LEFT_A, pixel_distance_RIGHT_A, pixel_distance_UP_A]) == 1 and pixel_value_RIGHT_A != 255:
                            valid_wall_lines.append([[X1, Y1, X1 + pixel_distance_RIGHT_A, Y1]])
                            is_extended = True
                        if np.argmin([pixel_distance_LEFT_A, pixel_distance_RIGHT_A, pixel_distance_UP_A]) == 2 and pixel_value_UP_A != 255:
                            valid_wall_lines.append([[X1, Y1 - pixel_distance_UP_A, X1, Y1]])
                            is_extended = True
                    pixel_distance_LEFT_B = np.argmin(canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, :X2].mean(axis=0)[::-1])
                    pixel_value_LEFT_B = canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, :X2].mean(axis=0)[::-1][np.argmin(canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, :X2].mean(axis=0)[::-1])]
                    pixel_distance_LEFT_B = pixel_distance_LEFT_B if pixel_value_LEFT_B != 255 else np.inf
                    pixel_distance_RIGHT_B = np.argmin(canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, X2 + 1:].mean(axis=0))
                    pixel_value_RIGHT_B = canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, X2 + 1:].mean(axis=0)[np.argmin(canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, X2 + 1:].mean(axis=0))]
                    pixel_distance_RIGHT_B = pixel_distance_RIGHT_B if pixel_value_RIGHT_B != 255 else np.inf
                    pixel_distance_DOWN_B = np.argmin(canvas[Y2 + 1:, X2 - open_tolerance_threshold:X2 + open_tolerance_threshold].mean(axis=1))
                    pixel_value_DOWN_B = canvas[Y2 + 1:, X2 - open_tolerance_threshold:X2 + open_tolerance_threshold].mean(axis=1)[np.argmin(canvas[Y2 + 1:, X2 - open_tolerance_threshold:X2 + open_tolerance_threshold].mean(axis=1))]
                    pixel_distance_DOWN_B = pixel_distance_DOWN_B if pixel_value_DOWN_B != 255 else np.inf
                    if min(pixel_distance_LEFT_B, pixel_distance_RIGHT_B, pixel_distance_DOWN_B) <= 500:
                        if np.argmin([pixel_distance_LEFT_B, pixel_distance_RIGHT_B, pixel_distance_DOWN_B]) == 0 and pixel_value_LEFT_B != 255:
                            valid_wall_lines.append([[X2 - pixel_distance_LEFT_B, Y2, X2, Y2]])
                            is_extended = True
                        if np.argmin([pixel_distance_LEFT_B, pixel_distance_RIGHT_B, pixel_distance_DOWN_B]) == 1 and pixel_value_RIGHT_B != 255:
                            valid_wall_lines.append([[X2, Y2, X2 + pixel_distance_RIGHT_B, Y2]])
                            is_extended = True
                        if np.argmin([pixel_distance_LEFT_B, pixel_distance_RIGHT_B, pixel_distance_DOWN_B]) == 2 and pixel_value_DOWN_B != 255:
                            valid_wall_lines.append([[X2, Y2, X2, Y2 + pixel_distance_DOWN_B]])
                            is_extended = True
                    if is_extended:
                        valid_wall_lines.append(wall_line)

                continue
            valid_wall_lines.append(wall_line)

        return valid_wall_lines

    def _deduplicate_lines(self, wall_lines, tolerance=10):
        if wall_lines is None:
            return wall_lines
        unique = list()
        for l in wall_lines:
            x1, y1, x2, y2 = l[0]
            duplicate = False
            for u in unique:
                ux1, uy1, ux2, uy2 = u[0]
                if (abs(x1-ux1) <= tolerance and abs(y1-uy1) <= tolerance and 
                    abs(x2-ux2) <= tolerance and abs(y2-uy2) <= tolerance):
                    duplicate = True
                    break
            if not duplicate:
                unique.append(l)
        horizontal_wall_lines = list()
        vertical_wall_lines = list()
        deduplicated = list()

        for wall_line in unique:
            X1, Y1, X2, Y2 = wall_line[0]
            line_type = self.classify_line(X1, Y1, X2, Y2)

            if line_type == "horizontal":
                horizontal_wall_lines.append(wall_line)
            elif line_type == "vertical":
                vertical_wall_lines.append(wall_line)
            else:
                deduplicated.append(wall_line)

        for horizontal_wall_line in horizontal_wall_lines:
            X1, Y1, X2, Y2 = horizontal_wall_line[0]
            target_horizontal_wall_lines = deepcopy(horizontal_wall_lines)
            target_horizontal_wall_lines.remove(horizontal_wall_line)
            is_duplicate = False
            for target_horizontal_wall_line in target_horizontal_wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_horizontal_wall_line[0]
                if X1 >= target_X1 and X2 <= target_X2 and abs(np.median([Y1, Y2]) - np.median([target_Y1, target_Y2])) <= tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(horizontal_wall_line)

        for vertical_wall_line in vertical_wall_lines:
            X1, Y1, X2, Y2 = vertical_wall_line[0]
            target_vertical_wall_lines = deepcopy(vertical_wall_lines)
            target_vertical_wall_lines.remove(vertical_wall_line)
            is_duplicate = False
            for target_vertical_wall_line in target_vertical_wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_vertical_wall_line[0]
                if Y1 >= target_Y1 and Y2 <= target_Y2 and abs(np.median([X1, X2]) - np.median([target_X1, target_X2])) <= tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(vertical_wall_line)

        return deduplicated

    def _draw_line(self, wall_line, canvas, overlay_enabled=False):
        x1, y1, x2, y2 = wall_line[0]['x'], wall_line[0]['y'], wall_line[1]['x'], wall_line[1]['y']
        line_width = 1
        if overlay_enabled:
            line_width = 3
        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
            line_width
        )

        return canvas

    def _patch_to_line(self, patch_GRAY, floor_plan_path, scale):
        lines = self._preprocessing(
            patch_GRAY,
            floor_plan_path,
            scale,
        )
        if lines is None:
            return

        lines = self._deduplicate_lines(lines)

        return lines

    @property
    def scale(self):
        return self._scale

    def _load_ceiling_height_and_scale(self, cropped_plans_BGR):
        def normalize_scale(scale):
            if scale.find(':') != -1:
                on_paper, real_world = scale.split(':')
            if scale.find('=') != -1:
                on_paper, real_world = scale.split('=')
            return f"{round(float(Fraction(on_paper.strip('`"'))), 2)}``:{real_world}"
        parts = list()
        for cropped_plan_BGR in cropped_plans_BGR:
            _, canvas_buffer_array = cv2.imencode(".png", cropped_plan_BGR)
            bytes_canvas = canvas_buffer_array.tobytes()
            parts.append(Part.from_data(data=bytes_canvas, mime_type="image/png"))
        query = Content(role="user", parts=parts)
        try:
            if self._is_cached["SCALE_AND_CEILING_HEIGHT_DETECTOR"]:
                response, ceiling_height_and_scale = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_metadata_extraction.generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=ScaleAndCeilingHeightDetectorResponse,
                )
            else:
                response, ceiling_height_and_scale = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_metadata_extraction(SCALE_AND_CEILING_HEIGHT_DETECTOR).generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=ScaleAndCeilingHeightDetectorResponse,
                )
            scale, ceiling_height = normalize_scale(response.scale), response.ceiling_height
            if scale:
                self._scale = scale
                ceiling_height_and_scale["scale"] = scale
            else:
                ceiling_height_and_scale["scale"] = self._scale
            if not ceiling_height:
                ceiling_height_and_scale["ceiling_height"] = self._height_in_feet
        except Exception as e:
            logging.info(f"SYSTEM: Standard Scale and Ceiling Height detection failed with error: {e}")
            ceiling_height_and_scale = dict(ceiling_height=self._height_in_feet, scale=self._scale)

        new_pixel_aspect_ratio_to_feet = self.compute_pixel_aspect_ratio(ceiling_height_and_scale["scale"], self._hyperparameters["pixel_aspect_ratio_to_feet"])
        self._hyperparameters["pixel_aspect_ratio_to_feet"] = new_pixel_aspect_ratio_to_feet
        self._hyperparameters["modelling"]["pixel_aspect_ratio"] = new_pixel_aspect_ratio_to_feet
        self._hyperparameters["modelling"]["height_in_feet"] = ceiling_height_and_scale["ceiling_height"]
        return ceiling_height_and_scale

    def _is_wall_valid(
        self,
        wall_line,
        drywall_polygons,
        floor_plan_path,
    ):
        canvas = cv2.imread(floor_plan_path)
        for drywall_polygon in drywall_polygons:
            canvas_to_overlay = canvas.copy()
            cv2.fillPoly(canvas_to_overlay, pts=[drywall_polygon], color=(0, 0, 255))
            canvas = cv2.addWeighted(canvas_to_overlay, 0.7, canvas, 0.3, 0)
        X1, Y1, X2, Y2 = wall_line[0]
        cv2.line(canvas, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
        _, canvas_buffer_array = cv2.imencode(".png", canvas)
        bytes_canvas = canvas_buffer_array.tobytes()
        wall_line_structured = dict(wall=dict(X1=int(X1), Y1=int(Y1), X2=int(X2), Y2=int(Y2)))
        query = Content(role="user", parts=[
            Part.from_text(json.dumps(wall_line_structured)),
            Part.from_data(data=bytes_canvas, mime_type="image/png")
        ])
        try:
            if self._is_cached["WALL_RECTIFIER"]:
                _, is_valid = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_wall_rectification.generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=WallRectifierResponse,
                )
            else:
                _, is_valid = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_wall_rectification(WALL_RECTIFIER).generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=WallRectifierResponse,
                )
            if is_valid["confidence"] > 0.9:
                return is_valid["is_valid"]
            return True
        except Exception as e:
            logging.warning(f"SYSTEM: Wall validator failed with error: {e}")
            return True

    def _model_polygon(
        self,
        vertices,
        walls,
        area_target,
        polygons_pts,
        floor_plan_path,
        transcription_block_with_centroids,
        walls_unnormalized,
        threshold=1000,
        tolerance=10,
        height_default=9.125,
    ):
        def verify_tolerance_distance(dimension_wall, wall_unnormalized, confidence_score):
            if dimension_wall["length"] and dimension_wall["width"] and confidence_score >= 0.9:
                dimension_wall["length"] = round(dimension_wall["length"], 2)
                dimension_wall["width"] = round(dimension_wall["width"], 2)
                return dimension_wall
            X1, Y1, X2, Y2 = wall_unnormalized[0]
            length_target = round(math.hypot(
                (X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"],
                (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]
            ), 2)
            length_predicted = dimension_wall["length"]
            if not length_predicted:
                dimension_wall["length"] = length_target
            else:
                dimension_wall["length"] = round(length_predicted, 2)
            if not dimension_wall["width"]:
                dimension_wall["width"] = self._width_in_feet
            else:
                dimension_wall["width"] = round(dimension_wall["width"], 2)
            if length_predicted and abs(length_target - length_predicted) > tolerance:
                dimension_wall["length"] = length_target

            return dimension_wall

        def verify_tolerance_area(area_polygon_predicted, area_polygon_target, confidence_score):
            if area_polygon_predicted and confidence_score >= 0.9:
                return area_polygon_predicted
            if area_polygon_predicted and abs(area_polygon_target - area_polygon_predicted) > tolerance ** 2:
                return area_polygon_target

            return area_polygon_predicted

        def verify_tolerance_height(height_predicted, confidence_score):
            if height_predicted and height_predicted != -1 and confidence_score >= 0.9:
                return height_predicted

            return height_default

        canvas = cv2.imread(floor_plan_path)
        vertices = np.array(vertices)
        canvas_to_overlay = canvas.copy()
        cv2.fillPoly(canvas_to_overlay, pts=[vertices], color=(0, 0, 255))
        canvas = cv2.addWeighted(canvas_to_overlay, 0.3, canvas, 0.7, 0)
        for wall in walls:
            X1, Y1, X2, Y2 = wall[0]
            bounding_box_top_left = (X1 - 10, Y1 - 10)
            bounding_box_bottom_right = (X2 + 10, Y2 + 10)
            canvas = cv2.rectangle(canvas, bounding_box_top_left, bounding_box_bottom_right, (255, 0, 0), 3)
        for polygon_pts in polygons_pts:
            canvas_to_overlay = canvas.copy()
            cv2.fillPoly(canvas_to_overlay, pts=[polygon_pts], color=(0, 255, 0))
            canvas = cv2.addWeighted(canvas_to_overlay, 0.5, canvas, 0.5, 0)
        polygon_bounding_box_X1 = min(vertex[0] for vertex in vertices)
        polygon_bounding_box_Y1 = min(vertex[1] for vertex in vertices)
        polygon_bounding_box_X2 = max(vertex[0] for vertex in vertices)
        polygon_bounding_box_Y2 = max(vertex[1] for vertex in vertices)
        threshold_X = max((polygon_bounding_box_X2 - polygon_bounding_box_X1) // 2, threshold)
        threshold_Y = max((polygon_bounding_box_Y2 - polygon_bounding_box_Y1) // 2, threshold)
        canvas_cropped = canvas[max(0, polygon_bounding_box_Y1 - threshold_Y): polygon_bounding_box_Y2 + threshold_Y, max(0, polygon_bounding_box_X1 - threshold_X): polygon_bounding_box_X2 + threshold_X]
        centroid_polygon_X = round(sum([vertex[0] for vertex in vertices]) / len(vertices))
        centroid_polygon_Y = round(sum([vertex[1] for vertex in vertices]) / len(vertices))
        nearest_transcription_blocks = self._load_nearest_transcription_blocks((centroid_polygon_X, centroid_polygon_Y), transcription_block_with_centroids)
        transcription_entries = list()
        for transcription, centroid in nearest_transcription_blocks.items():
            transcription_entries.append(dict(text=transcription, centroid=dict(X=centroid[0], Y=centroid[1])))
        _, canvas_buffer_array = cv2.imencode(".png", canvas_cropped)
        bytes_canvas = canvas_buffer_array.tobytes()
        # Pre-compute dimension candidates for each wall
        pixel_aspect_ratio = self._hyperparameters["modelling"]["pixel_aspect_ratio"]

        wall_specs = list()
        for wall in walls:
            X1, Y1, X2, Y2 = wall[0]
            candidates = extract_wall_dimension_candidates(
                [X1, Y1, X2, Y2], nearest_transcription_blocks, pixel_aspect_ratio
            )
            wall_spec = dict(
                wall=dict(X1=int(X1), Y1=int(Y1), X2=int(X2), Y2=int(Y2)),
                dimension_candidates=candidates
            )
            wall_specs.append(wall_spec)
        total_candidates = sum(len(ws["dimension_candidates"]) for ws in wall_specs)
        high_confidence = sum(1 for ws in wall_specs if ws["dimension_candidates"] and ws["dimension_candidates"][0]["confidence"] == "high")
        log_json("INFO", "DIMENSION_PRECOMPUTE",
                 total_walls=len(wall_specs),
                 total_candidates=total_candidates,
                 high_confidence_walls=high_confidence)
        polygon = dict(vertices=vertices.tolist(), perimeter_wall_lines=wall_specs, transcription_entries=transcription_entries)
        query = Content(role="user", parts=[
            Part.from_text(json.dumps(polygon)),
            Part.from_data(data=bytes_canvas, mime_type="image/png")
        ])
        #query_json = dict(
        #    user_parts=[
        #        dict(mime_type="text/plain", content=json.dumps(polygon)),
        #        dict(mime_type="image/png", content=b64encode(bytes_canvas).decode())
        #    ]
        #)
        #status_code, model_polygon = polygon_to_structured_2d(self._credentials, query_json)
        #if status_code == 200:
        try:
            if self._is_cached["DRYWALL_PREDICTOR_CALIFORNIA"]:
                _, model_polygon = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_drywall_prediction.generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=DrywallPredictorCaliforniaResponse,
                    verify_field_counts=dict(wall_parameters=len(perimeter_lines)),
                )
            else:
                _, model_polygon = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_drywall_prediction(DRYWALL_PREDICTOR_CALIFORNIA).generate_content(
                        contents=[feedback_prompt, query] if feedback_prompt else [query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=DrywallPredictorCaliforniaResponse,
                    verify_field_counts=dict(wall_parameters=len(perimeter_lines)),
                )
            model_polygon["ceiling"]["area"] = verify_tolerance_area(model_polygon["ceiling"]["area"], area_target, model_polygon["ceiling"]["confidence_area"])
            model_polygon["ceiling"]["height"] = verify_tolerance_height(model_polygon["ceiling"]["height"], model_polygon["ceiling"]["confidence_height"])
            for index, (dimension_wall_predicted, wall_unnormalized )in enumerate(zip(model_polygon["wall_parameters"], walls_unnormalized)):
                dimension_wall_rectified = verify_tolerance_distance(dimension_wall_predicted, wall_unnormalized, dimension_wall_predicted["confidence_length"])
                dimension_wall_rectified["height"] = verify_tolerance_height(dimension_wall_predicted["height"], dimension_wall_predicted["confidence_height"])
                model_polygon["wall_parameters"][index] = dimension_wall_rectified
                log_json("INFO", "DRYWALL_PREDICTION_SUCCESS",
                     room_name=model_polygon.get("ceiling", {}).get("room_name", ""),
                     n_walls=len(model_polygon.get("wall_parameters", [])))
        except Exception as e:
        #else:
            log_json("WARNING", "DRYWALL_PREDICTION_FAILED",
                     error=str(e),
                     n_walls=len(walls),
                     n_transcription_entries=len(transcription_entries),
                     n_dimension_candidates=sum(len(ws.get("dimension_candidates", [])) for ws in wall_specs))
            #logging.warning(f"SYSTEM: Drywall prediction for polygon: {json.dumps(polygon)} failed")
            model_polygon = {
                "ceiling": {
                    "room_name": '',
                    "area": area_target,
                    "ceiling_type": "Flat",
                    "height": height_default,
                    "slope": 0,
                    "slope_enabled": False,
                    "tilt_axis": '',
                    "drywall_assembly": {
                        "material": "D12C - 1/2\" DW INTERIOR CEILING",
                        "color_code": [10, 78, 69],
                        "thickness": 0.04,
                        "layers": 1,
                        "fire_rating": 0,
                        "waste_factor": "8-12%",
                    },
                    "code_references": list(),
                    "recommendation": ''
                }
            }
            wall_parameters = list()
            for wall in walls:
                X1, Y1, X2, Y2 = wall[0]
                length = round(math.hypot(
                    (X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"],
                    (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]
                ), 2)
                wall_parameters.append(
                    {
                        "room_name": '',
                        "length": length,
                        "width": self._hyperparameters["modelling"]["width_in_feet"],
                        "height": height_default,
                        "wall_type": '',
                        "drywall_assembly": {
                            "material": "D12L - 1/2\" DW LITE-WEIGHT",
                            "color_code": [71, 239, 143],
                            "materials_vertically_stacked": [],
                            "color_codes_stacked": [],
                            "thickness": 0.04,
                            "layers": 1,
                            "fire_rating": 0,
                            "waste_factor": "8-12%"
                        },
                        "code_references": list(),
                        "recommendation": ''
                    }
                )
            model_polygon["wall_parameters"] = wall_parameters

        return model_polygon

    def _add_walls_polygon(
        self,
        vertices,
        area,
        perimeter_walls,
        polygons,
        scale,
        height_default,
        floor_plan_path,
        transcription_block_with_centroids,
        index,
    ):
        def load_wall_payload(wall_line):
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line_structured = [
                dict(x=int(X1), y=int(Y1)),
                dict(x=int(X2), y=int(Y2))
            ]
            for wall_2d in self._walls_2d:
                if wall_2d["wall_line"] == wall_line_structured:
                    return wall_2d

        scale_x, scale_y = scale
        perimeter_walls_unnormalized = list()
        for perimeter_wall in perimeter_walls:
            X1, Y1, X2, Y2 = perimeter_wall[0]
            perimeter_wall_unnormalized = [[round(X1 / scale_x), round(Y1 / scale_y), round(X2 / scale_x), round(Y2 / scale_y)]]
            perimeter_walls_unnormalized.append(perimeter_wall_unnormalized)
        polygons_pts_normalized = list()
        for polygon in polygons:
            if isinstance(polygon, list):
                polygon = polygon[0]
            pts_normalized = np.array([
                [polygon["coordinates"][0]['x'], polygon["coordinates"][0]['y']],
                [polygon["coordinates"][1]['x'], polygon["coordinates"][1]['y']],
                [polygon["coordinates"][2]['x'], polygon["coordinates"][2]['y']],
                [polygon["coordinates"][3]['x'], polygon["coordinates"][3]['y']]
            ], np.int32)
            polygons_pts_normalized.append(pts_normalized)
        model_polygon = self._model_polygon(
            vertices,
            perimeter_walls,
            area,
            polygons_pts_normalized,
            floor_plan_path,
            transcription_block_with_centroids,
            perimeter_walls_unnormalized,
            height_default=height_default,
        )

        polygon_ids_drywall_interior = list()
        for wall_line, wall_parameter, polygon in zip(perimeter_walls, model_polygon["wall_parameters"], polygons):
            wall_payload = load_wall_payload(wall_line)
            if wall_payload:
                try:
                    thickness = round(float(wall_parameter["drywall_assembly"]["thickness"]), 2)
                except ValueError:
                    thickness = wall_parameter["drywall_assembly"]["thickness"]
                except (KeyError, TypeError):
                    wall_parameter["drywall_assembly"] = dict(
                        material="DISABLED",
                        color_code=[0, 0, 255],
                        materials_vertically_stacked=[],
                        color_codes_stacked=[],
                        thickness=-1,
                        layers=0,
                        fire_rating=0,
                        waste_factor="NA"
                    )
                    wall_parameter["recommendation"] = "NA"
                    wall_parameter["room_name"] = ''
                    thickness=-1
                if len(wall_payload["polygons_drywall"]) == 2:
                    continue
                wall_payload["polygons_drywall"].append(
                    dict(
                        id=f"{wall_payload["id"]}.b",
                        room_name=wall_parameter["room_name"],
                        polygon=polygon["coordinates"] if isinstance(polygon, dict) else polygon[0]["coordinates"],
                        type=wall_parameter["drywall_assembly"]["material"],
                        color=list(wall_parameter["drywall_assembly"]["color_code"]),
                        type_stacked=wall_parameter["drywall_assembly"]["materials_vertically_stacked"],
                        color_stacked=list(wall_parameter["drywall_assembly"]["color_codes_stacked"]),
                        thickness=thickness,
                        layers=wall_parameter["drywall_assembly"]["layers"],
                        fire_rating=wall_parameter["drywall_assembly"]["fire_rating"],
                        recommendation=wall_parameter["recommendation"],
                        waste_factor=wall_parameter["drywall_assembly"]["waste_factor"],
                        enabled=True
                    )
                )
                polygon_ids_drywall_interior.append(f"{wall_payload["id"]}.b")
            else:
                X1, Y1, X2, Y2 = wall_line[0]
                wall = dict(
                    id=len(self._walls_2d),
                    wall_line=[
                        dict(x=int(X1), y=int(Y1)),
                        dict(x=int(X2), y=int(Y2))
                    ],
                    thickness=wall_parameter["width"],
                    height=wall_parameter["height"] if wall_parameter["height"] else height_default,
                    length=wall_parameter["length"],
                    type=wall_parameter["wall_type"],
                    polygons_drywall=list()
                )
                try:
                    thickness = round(float(wall_parameter["drywall_assembly"]["thickness"]), 2)
                except ValueError:
                    thickness = wall_parameter["drywall_assembly"]["thickness"]
                except (KeyError, TypeError):
                    wall_parameter["drywall_assembly"] = dict(
                        material="DISABLED",
                        color_code=[0, 0, 255],
                        materials_vertically_stacked=[],
                        color_codes_stacked=[],
                        thickness=-1,
                        layers=0,
                        fire_rating=0,
                        waste_factor="NA"
                    )
                    wall_parameter["recommendation"] = "NA"
                    wall_parameter["room_name"] = ''
                    thickness=-1
                wall["polygons_drywall"].append(
                    dict(
                        id=f"{len(self._walls_2d)}.a",
                        room_name=wall_parameter["room_name"],
                        polygon=polygon["coordinates"] if isinstance(polygon, dict) else polygon[0]["coordinates"],
                        type=wall_parameter["drywall_assembly"]["material"],
                        color=list(wall_parameter["drywall_assembly"]["color_code"]),
                        type_stacked=wall_parameter["drywall_assembly"]["materials_vertically_stacked"],
                        color_stacked=list(wall_parameter["drywall_assembly"]["color_codes_stacked"]),
                        thickness=thickness,
                        layers=wall_parameter["drywall_assembly"]["layers"],
                        fire_rating=wall_parameter["drywall_assembly"]["fire_rating"],
                        recommendation=wall_parameter["recommendation"],
                        waste_factor=wall_parameter["drywall_assembly"]["waste_factor"],
                        enabled=True,
                    )
                )
                polygon_ids_drywall_interior.append(f"{len(self._walls_2d)}.a")
                if isinstance(polygon, list):
                    wall["polygons_drywall"].append(
                        dict(
                            id=f"{len(self._walls_2d)}.b",
                            room_name=wall_parameter["room_name"],
                            polygon=polygon[1]["coordinates"],
                            type=wall_parameter["drywall_assembly"]["material"],
                            color=list(wall_parameter["drywall_assembly"]["color_code"]),
                            type_stacked=wall_parameter["drywall_assembly"]["materials_vertically_stacked"],
                            color_stacked=list(wall_parameter["drywall_assembly"]["color_codes_stacked"]),
                            thickness=thickness,
                            layers=wall_parameter["drywall_assembly"]["layers"],
                            fire_rating=wall_parameter["drywall_assembly"]["fire_rating"],
                            recommendation=wall_parameter["recommendation"],
                            waste_factor=wall_parameter["drywall_assembly"]["waste_factor"],
                            enabled=True,
                        )
                    )
                    polygon_ids_drywall_interior.append(f"{len(self._walls_2d)}.b")
                self._walls_2d.append(wall)

        polygon_ids_drywall_interior_filtered = list()
        interior_wall_ids = set()
        for polygon_id_drywall_interior in polygon_ids_drywall_interior:
            wall_id = polygon_id_drywall_interior.split('.')[0]
            if wall_id in interior_wall_ids:
                continue
            polygon_ids_drywall_interior_filtered.append(polygon_id_drywall_interior)
            interior_wall_ids.add(wall_id)

        polygon = dict(
            id=index,
            area=model_polygon["ceiling"]["area"],
            vertices=vertices,
            type=model_polygon["ceiling"]["ceiling_type"],
            height=model_polygon["ceiling"]["height"] if model_polygon["ceiling"]["height"] else height_default,
            slope=0,
            slope_enabled=model_polygon["ceiling"]["slope_enabled"],
            tilt_axis=model_polygon["ceiling"]["tilt_axis"],
            room_name=model_polygon["ceiling"]["room_name"],
            polygon_ids_drywall_interior=polygon_ids_drywall_interior_filtered,
            polygon_drywall=dict(
                type=model_polygon["ceiling"]["drywall_assembly"]["material"],
                color=tuple(model_polygon["ceiling"]["drywall_assembly"]["color_code"]),
                thickness=model_polygon["ceiling"]["drywall_assembly"]["thickness"],
                layers=model_polygon["ceiling"]["drywall_assembly"]["layers"],
                fire_rating=model_polygon["ceiling"]["drywall_assembly"]["fire_rating"],
                recommendation=model_polygon["ceiling"]["recommendation"],
                waste_factor=model_polygon["ceiling"]["drywall_assembly"]["waste_factor"],
                enabled=True,
            )
        )
        self._polygons.append(polygon)

    def _add_wall_perimeter(
        self,
        wall_line,
        polygons,
        height_default,
        scale,
        thickness_default=0.29,
    ):
        X1, Y1, X2, Y2 = wall_line[0]
        wall_length_expected = round(math.hypot(
            (X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"],
            (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]
        ), 2)
        scale_x, scale_y = scale
        wall_line_structured = [
            dict(x=round(scale_x * X1), y=round(scale_y * Y1)),
            dict(x=round(scale_x * X2), y=round(scale_y * Y2))
        ]
        wall_payload = None
        for wall_2d in self._walls_2d:
            if wall_2d["wall_line"] == wall_line_structured:
                wall_payload = wall_2d

        if wall_payload:
            wall_payload["polygons_drywall"].append(
                dict(
                    id=f"{wall_payload["id"]}.b",
                    polygon=polygons[0]["coordinates"],
                    type="DISABLED",
                    color=[0, 0, 255],
                    type_stacked=[],
                    color_stacked=[],
                    thickness=-1,
                    layers=0,
                    fire_rating=0,
                    recommendation="NA",
                    waste_factor="NA",
                    enabled=False,
                    room_name='',
                )
            )
        else:
            wall = dict(
                id=len(self._walls_2d),
                wall_line=[
                    dict(x=round(scale_x * X1), y=round(scale_y * Y1)),
                    dict(x=round(scale_x * X2), y=round(scale_y * Y2))
                ],
                thickness=thickness_default,
                height=height_default,
                length=wall_length_expected,
                polygons_drywall=list(),
                type=''
            )
            for polygon, polygon_index in zip(polygons, ['a', 'b']):
                wall["polygons_drywall"].append(
                    dict(
                        id=f"{len(self._walls_2d)}.{polygon_index}",
                        polygon=polygon["coordinates"],
                        type="DISABLED",
                        color=[0, 0, 255],
                        type_stacked=[],
                        color_stacked=[],
                        thickness=-1,
                        layers=0,
                        fire_rating=0,
                        recommendation="NA",
                        waste_factor="NA",
                        enabled=False,
                        room_name='',
                    )
                )
            self._walls_2d.append(wall)

    def _extrude_polygon_perimeter(self, line, scale, outer_drywall_surface=None):
        polygons = list()
        scale_x, scale_y = scale
        X1_unnormalized, Y1_unnormalized, X2_unnormalized, Y2_unnormalized = line[0][0], line[0][1], line[0][2], line[0][3]
        orientation = self.classify_line(X1_unnormalized, Y1_unnormalized, X2_unnormalized, Y2_unnormalized)
        X1, Y1, X2, Y2 = round(scale_x * line[0][0]), round(scale_y * line[0][1]), round(scale_x * line[0][2]), round(scale_y * line[0][3])
        if orientation == "horizontal":
            polygon_a = [
                dict(x=X1+20, y=Y1-20),
                dict(x=X2-20, y=Y2-20),
                dict(x=X2-60, y=Y2-60),
                dict(x=X1+60, y=Y1-60)
            ]
            polygon_b = [
                dict(x=X1+20, y=Y1+20),
                dict(x=X2-20, y=Y2+20),
                dict(x=X2-60, y=Y2+60),
                dict(x=X1+60, y=Y1+60)
            ]
            if outer_drywall_surface == "UP" or outer_drywall_surface == "INVALID":
                polygons.append(dict(coordinates=polygon_a, enabled=False))
            if outer_drywall_surface == "DOWN" or outer_drywall_surface == "INVALID":
                polygons.append(dict(coordinates=polygon_b, enabled=False))

        if orientation == "vertical":
            polygon_a = [
                dict(x=X1-20, y=Y1+20),
                dict(x=X2-20, y=Y2-20),
                dict(x=X2-60, y=Y2-60),
                dict(x=X1-60, y=Y1+60)
            ]
            polygon_b = [
                dict(x=X1+20, y=Y1+20),
                dict(x=X2+20, y=Y2-20),
                dict(x=X2+60, y=Y2-60),
                dict(x=X1+60, y=Y1+60)
            ]
            if outer_drywall_surface == "LEFT" or outer_drywall_surface == "INVALID":
                polygons.append(dict(coordinates=polygon_a, enabled=False))
            if outer_drywall_surface == "RIGHT" or outer_drywall_surface == "INVALID":
                polygons.append(dict(coordinates=polygon_b, enabled=False))

        if orientation == "inclined":
            dx = X2 - X1
            dy = Y2 - Y1
            length = math.hypot(dx, dy)
            if length == 0:
                return polygons

            tx = dx / length
            ty = dy / length

            nx = -dy / length
            ny =  dx / length

            polygon_a = [
                dict(
                    x=int(X1 + nx * 20 + tx * 20),
                    y=int(Y1 + ny * 20 + ty * 20),
                ),
                dict(
                    x=int(X2 + nx * 20 - tx * 20),
                    y=int(Y2 + ny * 20 - ty * 20),
                ),
                dict(
                    x=int(X2 + nx * 60 - tx * 60),
                    y=int(Y2 + ny * 60 - ty * 60),
                ),
                dict(
                    x=int(X1 + nx * 60 + tx * 60),
                    y=int(Y1 + ny * 60 + ty * 60),
                ),
            ]
            polygons.append(dict(coordinates=polygon_a, enabled=False))

            polygon_b = [
                dict(
                    x=int(X1 - nx * 20 + tx * 20),
                    y=int(Y1 - ny * 20 + ty * 20),
                ),
                dict(
                    x=int(X2 - nx * 20 - tx * 20),
                    y=int(Y2 - ny * 20 - ty * 20),
                ),
                dict(
                    x=int(X2 - nx * 60 - tx * 60),
                    y=int(Y2 - ny * 60 - ty * 60),
                ),
                dict(
                    x=int(X1 - nx * 60 + tx * 60),
                    y=int(Y1 - ny * 60 + ty * 60),
                ),
            ]
            polygons.append(dict(coordinates=polygon_b, enabled=False))

        return polygons

    def _extrude_polygon_drywalls(self, polygon_perimeter_lines, polygon_vertices, scale):
        polygons = list()
        scale_x, scale_y = scale
        for polygon_perimeter_line in polygon_perimeter_lines:
            X1, Y1, X2, Y2 = polygon_perimeter_line[0][0], polygon_perimeter_line[0][1], polygon_perimeter_line[0][2], polygon_perimeter_line[0][3]
            orientation = self.classify_line(round(X1 / scale_x), round(Y1 / scale_y), round(X2 / scale_x), round(Y2 / scale_y))
            if orientation == "horizontal":
                centroid_perimeter_line = (round((X1 + X2) / 2), round(np.median([Y1, Y2])))
                polygon_up = [
                    dict(x=X1+20, y=Y1-20),
                    dict(x=X2-20, y=Y2-20),
                    dict(x=X2-60, y=Y2-60),
                    dict(x=X1+60, y=Y1-60)
                ]
                polygon_down = [
                    dict(x=X1+20, y=Y1+20),
                    dict(x=X2-20, y=Y2+20),
                    dict(x=X2-60, y=Y2+60),
                    dict(x=X1+60, y=Y1+60)
                ]
                if self.is_inside_polygon((centroid_perimeter_line[0], centroid_perimeter_line[1] - 50), polygon_vertices) and self.is_inside_polygon((centroid_perimeter_line[0], centroid_perimeter_line[1] + 50), polygon_vertices):
                    polygons.append([dict(coordinates=polygon_up, enabled=True), dict(coordinates=polygon_down, enabled=True)])
                elif self.is_inside_polygon((centroid_perimeter_line[0], centroid_perimeter_line[1] - 50), polygon_vertices):
                    polygons.append(dict(coordinates=polygon_up, enabled=True))
                else:
                    polygons.append(dict(coordinates=polygon_down, enabled=True))

            if orientation == "vertical":
                centroid_perimeter_line = (round(np.median([X1, X2])), round((Y1 + Y2) / 2))
                polygon_left = [
                        dict(x=X1-20, y=Y1+20),
                        dict(x=X2-20, y=Y2-20),
                        dict(x=X2-60, y=Y2-60),
                        dict(x=X1-60, y=Y1+60)
                    ]
                polygon_right = [
                    dict(x=X1+20, y=Y1+20),
                    dict(x=X2+20, y=Y2-20),
                    dict(x=X2+60, y=Y2-60),
                    dict(x=X1+60, y=Y1+60)
                ]
                if self.is_inside_polygon((centroid_perimeter_line[0] - 50, centroid_perimeter_line[1]), polygon_vertices) and self.is_inside_polygon((centroid_perimeter_line[0] + 50, centroid_perimeter_line[1]), polygon_vertices):
                    polygons.append([dict(coordinates=polygon_left, enabled=True), dict(coordinates=polygon_right, enabled=True)])
                elif self.is_inside_polygon((centroid_perimeter_line[0] - 50, centroid_perimeter_line[1]), polygon_vertices):
                    polygons.append(dict(coordinates=polygon_left, enabled=True))
                else:
                    polygons.append(dict(coordinates=polygon_right, enabled=True))

            if orientation == "inclined":
                dx = X2 - X1
                dy = Y2 - Y1
                length = math.hypot(dx, dy)
                if length == 0:
                    return polygons

                tx = dx / length
                ty = dy / length

                nx = -dy / length
                ny =  dx / length

                mx = (X1 + X2) / 2
                my = (Y1 + Y2) / 2

                test_coordinate_A = (
                    round(mx + nx * 50),
                    round(my + ny * 50)
                )
                test_coordinate_B = (
                    round(mx - nx * 50),
                    round(my - ny * 50)
                )

                polygon_A = [
                    dict(x=int(X1 + nx * 20 + tx * 20), y=int(Y1 + ny * 20 + ty * 20)),
                    dict(x=int(X2 + nx * 20 - tx * 20), y=int(Y2 + ny * 20 - ty * 20)),
                    dict(x=int(X2 + nx * 60 - tx * 60), y=int(Y2 + ny * 60 - ty * 60)),
                    dict(x=int(X1 + nx * 60 + tx * 60), y=int(Y1 + ny * 60 + ty * 60)),
                ]
                polygon_B = [
                    dict(x=int(X1 - nx * 20 + tx * 20), y=int(Y1 - ny * 20 + ty * 20)),
                    dict(x=int(X2 - nx * 20 - tx * 20), y=int(Y2 - ny * 20 - ty * 20)),
                    dict(x=int(X2 - nx * 60 - tx * 60), y=int(Y2 - ny * 60 - ty * 60)),
                    dict(x=int(X1 - nx * 60 + tx * 60), y=int(Y1 - ny * 60 + ty * 60)),
                ]

                if self.is_inside_polygon(test_coordinate_A, polygon_vertices) and self.is_inside_polygon(test_coordinate_B, polygon_vertices):
                    polygons.append([dict(coordinates=polygon_A, enabled=True), dict(coordinates=polygon_B, enabled=True)])
                elif self.is_inside_polygon(test_coordinate_A, polygon_vertices):
                    polygons.append(dict(coordinates=polygon_A, enabled=True))
                else:
                    polygons.append(dict(coordinates=polygon_B, enabled=True))

        return polygons

    def _normalize_walls_2d(
        self,
        walls_2d,
        scale,
        remove_drywall_disabled=False,
        impute_drywall_disabled=False,
        polygon_vertices_external=None
    ):
        scale_x, scale_y = scale
        for wall in walls_2d[:]:
            if impute_drywall_disabled and len(wall["polygons_drywall"]) == 2:
                if not wall["polygons_drywall"][0]["enabled"] or not wall["polygons_drywall"][1]["enabled"]:
                    centroid_A = (round(sum([vertex['x'] for vertex in wall["polygons_drywall"][0]["polygon"]]) / 4), round(sum([vertex['y'] for vertex in wall["polygons_drywall"][0]["polygon"]]) / 4))
                    centroid_B = (round(sum([vertex['x'] for vertex in wall["polygons_drywall"][1]["polygon"]]) / 4), round(sum([vertex['y'] for vertex in wall["polygons_drywall"][1]["polygon"]]) / 4))
                    reference_line = [[wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']]]
                    reference_line_unnormalized = [round(wall["wall_line"][0]['x'] / scale_x), round(wall["wall_line"][0]['y'] / scale_y), round(wall["wall_line"][1]['x'] / scale_x), round(wall["wall_line"][1]['y'] / scale_y)]
                    if self.is_inside_polygon(centroid_A, polygon_vertices_external) and self.is_inside_polygon(centroid_B, polygon_vertices_external):
                        target_lines = [[[wall_target["wall_line"][0]['x'], wall_target["wall_line"][0]['y'], wall_target["wall_line"][1]['x'], wall_target["wall_line"][1]['y']]] for wall_target in walls_2d[:]]
                        neighbors = self.nearest_neighbor(reference_line, 'A', target_lines, top_k=5)
                        valid_neighbor_found = False
                        for neighbor in neighbors:
                            neighbor_unnormalized = [round(neighbor[0][0] / scale_x), round(neighbor[0][1] / scale_y), round(neighbor[0][2] / scale_x), round(neighbor[0][3] / scale_y)]
                            for wall_ in walls_2d:
                                if neighbor == [[wall_["wall_line"][0]['x'], wall_["wall_line"][0]['y'], wall_["wall_line"][1]['x'], wall_["wall_line"][1]['y']]] and (wall_["polygons_drywall"][0]["enabled"] or wall_["polygons_drywall"][1]["enabled"]):
                                    if self.classify_line(*reference_line_unnormalized) == self.classify_line(*neighbor_unnormalized):
                                        if not wall["polygons_drywall"][0]["enabled"]:
                                            wall["polygons_drywall"][0]["color"] = wall_["polygons_drywall"][0]["color"]
                                            wall["polygons_drywall"][0]["enabled"] = wall_["polygons_drywall"][0]["enabled"]
                                            wall["polygons_drywall"][0]["fire_rating"] = wall_["polygons_drywall"][0]["fire_rating"]
                                            wall["polygons_drywall"][0]["layers"] = wall_["polygons_drywall"][0]["layers"]
                                            wall["polygons_drywall"][0]["thickness"] = wall_["polygons_drywall"][0]["thickness"]
                                            wall["polygons_drywall"][0]["type"] = wall_["polygons_drywall"][0]["type"]
                                            wall["polygons_drywall"][0]["waste_factor"] = wall_["polygons_drywall"][0]["waste_factor"]
                                            wall["polygons_drywall"][0]["recommendation"] = f"FP - Drywall material has been imputed from the nearby wall from Room: {wall_["polygons_drywall"][0]["room_name"]}."
                                        if not wall["polygons_drywall"][1]["enabled"]:
                                            wall["polygons_drywall"][1]["color"] = wall_["polygons_drywall"][1]["color"]
                                            wall["polygons_drywall"][1]["enabled"] = wall_["polygons_drywall"][1]["enabled"]
                                            wall["polygons_drywall"][1]["fire_rating"] = wall_["polygons_drywall"][1]["fire_rating"]
                                            wall["polygons_drywall"][1]["layers"] = wall_["polygons_drywall"][1]["layers"]
                                            wall["polygons_drywall"][1]["thickness"] = wall_["polygons_drywall"][1]["thickness"]
                                            wall["polygons_drywall"][1]["type"] = wall_["polygons_drywall"][1]["type"]
                                            wall["polygons_drywall"][1]["waste_factor"] = wall_["polygons_drywall"][1]["waste_factor"]
                                            wall["polygons_drywall"][1]["recommendation"] = f"FP - Drywall material has been imputed from nearby wall from Room: {wall_["polygons_drywall"][1]["room_name"]}."
                                    else:
                                        if not wall["polygons_drywall"][0]["enabled"]:
                                            wall["polygons_drywall"][0]["color"] = wall_["polygons_drywall"][1]["color"]
                                            wall["polygons_drywall"][0]["enabled"] = wall_["polygons_drywall"][1]["enabled"]
                                            wall["polygons_drywall"][0]["fire_rating"] = wall_["polygons_drywall"][1]["fire_rating"]
                                            wall["polygons_drywall"][0]["layers"] = wall_["polygons_drywall"][1]["layers"]
                                            wall["polygons_drywall"][0]["thickness"] = wall_["polygons_drywall"][1]["thickness"]
                                            wall["polygons_drywall"][0]["type"] = wall_["polygons_drywall"][1]["type"]
                                            wall["polygons_drywall"][0]["waste_factor"] = wall_["polygons_drywall"][1]["waste_factor"]
                                            wall["polygons_drywall"][0]["recommendation"] = f"FP - Drywall material has been imputed from the nearby wall from Room: {wall_["polygons_drywall"][1]["room_name"]}."
                                        if not wall["polygons_drywall"][1]["enabled"]:
                                            wall["polygons_drywall"][1]["color"] = wall_["polygons_drywall"][0]["color"]
                                            wall["polygons_drywall"][1]["enabled"] = wall_["polygons_drywall"][0]["enabled"]
                                            wall["polygons_drywall"][1]["fire_rating"] = wall_["polygons_drywall"][0]["fire_rating"]
                                            wall["polygons_drywall"][1]["layers"] = wall_["polygons_drywall"][0]["layers"]
                                            wall["polygons_drywall"][1]["thickness"] = wall_["polygons_drywall"][0]["thickness"]
                                            wall["polygons_drywall"][1]["type"] = wall_["polygons_drywall"][0]["type"]
                                            wall["polygons_drywall"][1]["waste_factor"] = wall_["polygons_drywall"][0]["waste_factor"]
                                            wall["polygons_drywall"][1]["recommendation"] = f"FP - Drywall material has been imputed from nearby wall from Room: {wall_["polygons_drywall"][0]["room_name"]}."
                                    valid_neighbor_found = True
                                    break
                            if valid_neighbor_found:
                                break
            if remove_drywall_disabled:
                if not wall["polygons_drywall"] or (not wall["polygons_drywall"][0]["enabled"] and not wall["polygons_drywall"][1]["enabled"]):
                    walls_2d.remove(wall)
                    continue
            if len(wall["polygons_drywall"]) == 2 and wall["polygons_drywall"][0]["polygon"] != wall["polygons_drywall"][1]["polygon"]:
                continue
            wall_line_vertices = wall["wall_line"]
            X1, Y1, X2, Y2 = wall_line_vertices[0]['x'], wall_line_vertices[0]['y'], wall_line_vertices[1]['x'], wall_line_vertices[1]['y']
            orientation = self.classify_line(round(X1 / scale_x), round(Y1 / scale_y), round(X2 / scale_x), round(Y2 / scale_y))
            if not wall["polygons_drywall"]:
                for index in ['a', 'b']:
                    if orientation == "horizontal":
                        if index == 'a':
                            polygon_vertices = [
                                dict(x=X1+20, y=Y1-20),
                                dict(x=X2-20, y=Y2-20),
                                dict(x=X2-60, y=Y2-60),
                                dict(x=X1+60, y=Y1-60)
                            ]
                        else:
                            polygon_vertices = [
                                dict(x=X1+20, y=Y1+20),
                                dict(x=X2-20, y=Y2+20),
                                dict(x=X2-60, y=Y2+60),
                                dict(x=X1+60, y=Y1+60)
                            ]
                    if orientation == "vertical":
                        if index == 'a':   
                            polygon_vertices = [
                                dict(x=X1-20, y=Y1+20),
                                dict(x=X2-20, y=Y2-20),
                                dict(x=X2-60, y=Y2-60),
                                dict(x=X1-60, y=Y1+60)
                            ]
                        else:
                            polygon_vertices = [
                                dict(x=X1+20, y=Y1+20),
                                dict(x=X2+20, y=Y2-20),
                                dict(x=X2+60, y=Y2-60),
                                dict(x=X1+60, y=Y1+60)
                            ]
                    if orientation == "inclined":
                        dx = X2 - X1
                        dy = Y2 - Y1
                        length = math.hypot(dx, dy)
                        if length == 0:
                            continue

                        tx = dx / length
                        ty = dy / length

                        nx = -dy / length
                        ny =  dx / length

                        if index == 'a':
                            polygon_vertices = [
                                dict(
                                    x=int(X1 + nx * 20 + tx * 20),
                                    y=int(Y1 + ny * 20 + ty * 20),
                                ),
                                dict(
                                    x=int(X2 + nx * 20 - tx * 20),
                                    y=int(Y2 + ny * 20 - ty * 20),
                                ),
                                dict(
                                    x=int(X2 + nx * 60 - tx * 60),
                                    y=int(Y2 + ny * 60 - ty * 60),
                                ),
                                dict(
                                    x=int(X1 + nx * 60 + tx * 60),
                                    y=int(Y1 + ny * 60 + ty * 60),
                                 ),
                                ]
                        else:
                            polygon_vertices = [
                                dict(
                                    x=int(X1 - nx * 20 + tx * 20),
                                    y=int(Y1 - ny * 20 + ty * 20),
                                ),
                                dict(
                                    x=int(X2 - nx * 20 - tx * 20),
                                    y=int(Y2 - ny * 20 - ty * 20),
                                ),
                                dict(
                                    x=int(X2 - nx * 60 - tx * 60),
                                    y=int(Y2 - ny * 60 - ty * 60),
                                ),
                                dict(
                                    x=int(X1 - nx * 60 + tx * 60),
                                    y=int(Y1 - ny * 60 + ty * 60),
                                 ),
                                ]
                    wall["polygons_drywall"].append(
                        dict(
                            id=f"{wall["id"]}.{index}",
                            room_name='',
                            polygon=polygon_vertices,
                            type="DISABLED",
                            color=[0, 0, 255],
                            type_stacked=[],
                            color_stacked=[],
                            thickness=-1,
                            layers=0,
                            fire_rating=0,
                            recommendation='',
                            waste_factor="NA",
                            enabled=False
                        )
                    )
            else:
                polygon_vertices = wall["polygons_drywall"][0]["polygon"]
                centroid_polygon_X = round(sum([vertex['x'] for vertex in polygon_vertices]) / 4)
                centroid_polygon_Y = round(sum([vertex['y'] for vertex in polygon_vertices]) / 4)
                if orientation == "horizontal":
                    if centroid_polygon_Y <= np.median([Y1, Y2]):
                        polygon_vertices = [
                            dict(x=X1+20, y=Y1+20),
                            dict(x=X2-20, y=Y2+20),
                            dict(x=X2-60, y=Y2+60),
                            dict(x=X1+60, y=Y1+60)
                        ]
                    else:
                        polygon_vertices = [
                            dict(x=X1+20, y=Y1-20),
                            dict(x=X2-20, y=Y2-20),
                            dict(x=X2-60, y=Y2-60),
                            dict(x=X1+60, y=Y1-60)
                        ]
                if orientation == "vertical":
                    if centroid_polygon_X <= np.median([X1, X2]):
                        polygon_vertices = [
                            dict(x=X1+20, y=Y1+20),
                            dict(x=X2+20, y=Y2-20),
                            dict(x=X2+60, y=Y2-60),
                            dict(x=X1+60, y=Y1+60)
                        ]
                    else:
                        polygon_vertices = [
                            dict(x=X1-20, y=Y1+20),
                            dict(x=X2-20, y=Y2-20),
                            dict(x=X2-60, y=Y2-60),
                            dict(x=X1-60, y=Y1+60)
                        ]
                if orientation == "inclined":
                    dx = X2 - X1
                    dy = Y2 - Y1
                    length = math.hypot(dx, dy)
                    if length == 0:
                        continue

                    tx = dx / length
                    ty = dy / length

                    nx = -dy / length
                    ny =  dx / length

                    mx = (X1 + X2) / 2
                    my = (Y1 + Y2) / 2

                    vx = centroid_polygon_X - mx
                    vy = centroid_polygon_Y - my

                    dot = nx * vx + ny * vy

                    if dot > 0:
                        nx, ny = -nx, -ny

                    polygon_vertices = [
                        dict(
                            x=int(X1 + nx * 20 + tx * 20),
                            y=int(Y1 + ny * 20 + ty * 20),
                        ),
                        dict(
                            x=int(X2 + nx * 20 - tx * 20),
                            y=int(Y2 + ny * 20 - ty * 20),
                        ),
                        dict(
                            x=int(X2 + nx * 60 - tx * 60),
                            y=int(Y2 + ny * 60 - ty * 60),
                        ),
                        dict(
                            x=int(X1 + nx * 60 + tx * 60),
                            y=int(Y1 + ny * 60 + ty * 60),
                        ),
                    ]
                if len(wall["polygons_drywall"]) == 2:
                    wall["polygons_drywall"].pop()
                wall["polygons_drywall"].append(
                    dict(
                        id=f"{wall["id"]}.b",
                        room_name='',
                        polygon=polygon_vertices,
                        type="DISABLED",
                        color=[0, 0, 255],
                        type_stacked=[],
                        color_stacked=[],
                        thickness=-1,
                        layers=0,
                        fire_rating=0,
                        recommendation='',
                        waste_factor="NA",
                        enabled=False
                    )
                )

        return walls_2d

    def _normalize_polygons(self, polygons, walls_2d):
        polygons_valid = list()
        walls_2d_ids = [wall["id"] for wall in walls_2d]
        for polygon in polygons:
            perimeter_wall_missing = False
            for drywall_id in polygon["polygon_ids_drywall_interior"]:
                wall_id = int(drywall_id.split('.')[0])
                if wall_id not in walls_2d_ids:
                    perimeter_wall_missing = True
                    break
            if not perimeter_wall_missing:
                polygons_valid.append(polygon)
        return polygons_valid

    @classmethod
    def scale_to(
        cls,
        floor_plan_path="/tmp/floor_plan.png",
        resolution=None
    ):
        canvas = Image.open(floor_plan_path)
        width_in_pixels, height_in_pixels = canvas.size
        if canvas.mode != "RGB":
            canvas = canvas.convert("RGB")

        if resolution:
            canvas = canvas.resize(resolution, Image.Resampling.LANCZOS)
            width_in_pixels, height_in_pixels = resolution

        pdf_path = "/tmp/scaled_floor_plan.pdf"
        canvas.save(pdf_path, save_all=True)

        svg_path = "/tmp/scaled_floor_plan.svg"
        subprocess.run(
            ["pdftocairo", "-svg", pdf_path, svg_path],
            check=True
        )
        tree = ET.parse(svg_path)
        root = tree.getroot()
        width_in_points = root.attrib.get("width")
        height_in_points = root.attrib.get("height")
        root.set("width", "100%")
        root.set("height", "100%")
        if not root.get("preserveAspectRatio"):
            root.set("preserveAspectRatio", "xMidYMid meet")
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)

        return Path(svg_path), dict(
            height_in_pixels=height_in_pixels,
            width_in_pixels=width_in_pixels,
            height_in_points=height_in_points,
            width_in_points=width_in_points,
            size=Path(svg_path).stat().st_size
        )

    def load_drywall_choices(self, walls_2d_JSON, polygons_2d_JSON):
        drywall_choices = ["DISABLED"] + [drywall_template["sku_variant"] for drywall_template in self._drywall_templates]
        for wall in walls_2d_JSON:
            wall["drywall_choices"] = drywall_choices
            for polygon_drywall in wall["polygons_drywall"]:
                polygon_drywall["color"] = polygon_drywall["color"][::-1]
        for polygon in polygons_2d_JSON:
            polygon["drywall_choices"] = drywall_choices
            polygon["polygon_drywall"]["color"] = polygon["polygon_drywall"]["color"][::-1]

    @property
    def drywall_choices_color_codes(self):
        drywall_choices_color_codes={drywall_template["sku_variant"]: drywall_template["color_code"][::-1] for drywall_template in self._drywall_templates}
        drywall_choices_color_codes.update(dict(DISABLED=[255, 0, 0]))
        return drywall_choices_color_codes

    def load_ceiling_choices(self, polygons_2d_JSON):
        for polygon in polygons_2d_JSON:
            polygon["type_choices"] = CEILING_CHOICES

    def _load_missing_polygons(self, walls_2d, scale, polygons_neighbor, floor_plan_path):
        def load_wall_payload(wall_line):
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line_structured = [
                dict(x=int(X1), y=int(Y1)),
                dict(x=int(X2), y=int(Y2))
            ]
            for wall_2d in walls_2d:
                if wall_2d["wall_line"] == wall_line_structured:
                    return wall_2d

        def load_polygon(shape, lines_isolated, tolerance=10):
            polygon_vertices = list()
            for wall_line in shape:
                X1, Y1, X2, Y2 = wall_line[0]
                if all([math.hypot(polygon_vertex[0] - X1, polygon_vertex[1] - Y1) > tolerance for polygon_vertex in polygon_vertices]):
                    polygon_vertices.append([round(X1), round(Y1)])
                if all([math.hypot(polygon_vertex[0] - X2, polygon_vertex[1] - Y2) > tolerance for polygon_vertex in polygon_vertices]):
                    polygon_vertices.append([round(X2), round(Y2)])
            for wall_line in lines_isolated:
                X1, Y1, X2, Y2 = wall_line[0]
                if all([math.hypot(polygon_vertex[0] - X1, polygon_vertex[1] - Y1) > tolerance for polygon_vertex in polygon_vertices]):
                    polygon_vertices.append([round(X1), round(Y1)])
                if all([math.hypot(polygon_vertex[0] - X2, polygon_vertex[1] - Y2) > tolerance for polygon_vertex in polygon_vertices]):
                    polygon_vertices.append([round(X2), round(Y2)])

            polygon_vertices_array = np.asarray(polygon_vertices, dtype=np.int32)
            centroid = polygon_vertices_array.mean(axis=0)
            vectors = polygon_vertices_array - centroid
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            distances = np.sum(vectors**2, axis=1)

            order = np.lexsort((distances, angles))
            polygon_vertices = polygon_vertices_array[order].tolist()

            vertices_sorted = deepcopy(polygon_vertices)
            for index in range(len(polygon_vertices)-2):
                reference = vertices_sorted[index]
                if math.hypot(vertices_sorted[index + 1][0] - reference[0], vertices_sorted[index + 1][1] - reference[1]) > math.hypot(vertices_sorted[index + 2][0] - reference[0], vertices_sorted[index + 2][1] - reference[1]):
                    sorted_vertex_a = vertices_sorted[index + 1]
                    sorted_vertex_b = vertices_sorted[index + 2]
                    vertices_sorted.remove(sorted_vertex_a)
                    vertices_sorted.remove(sorted_vertex_b)
                    vertices_sorted.insert(index + 1, sorted_vertex_b)
                    vertices_sorted.insert(index + 2, sorted_vertex_a)
            return vertices_sorted

        def is_valid(polygon_vertices, iou_threshold=0.25):
            polygon_target = Polygon(polygon_vertices)
            if not polygon_target.is_valid:
                return False
            for polygon in polygons_neighbor:
                try:
                    polygon_neighbor = Polygon(polygon)
                except ValueError:
                    continue
                if not polygon_neighbor.is_valid:
                    polygon_neighbor = polygon_neighbor.buffer(0)
                intersection = polygon_target.intersection(polygon_neighbor).area
                union = polygon_target.union(polygon_neighbor).area
                if union == 0:
                    return False
                iou = intersection / union
                if iou > iou_threshold:
                    return False
            return True

        walls_null_room, walls_null_id, is_valid_futures = list(), list(), list()
        with ThreadPoolExecutor(max_workers=4) as executor:
            for wall in walls_2d:
                if wall["polygons_drywall"][0]["type"] == wall["polygons_drywall"][1]["type"] == "DISABLED":
                    wall_line = [[wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']]]
                    drywall_polygons = list()
                    for drywall in wall["polygons_drywall"]:
                        pts = np.array([
                            [drywall["polygon"][0]['x'], drywall["polygon"][0]['y']],
                            [drywall["polygon"][1]['x'], drywall["polygon"][1]['y']],
                            [drywall["polygon"][2]['x'], drywall["polygon"][2]['y']],
                            [drywall["polygon"][3]['x'], drywall["polygon"][3]['y']]
                        ], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        drywall_polygons.append(pts)
                    is_valid_futures.append(executor.submit(
                        self._is_wall_valid,
                        wall_line,
                        drywall_polygons,
                        floor_plan_path,
                    ))
            is_wall_valid = [future.result() for future in is_valid_futures]
        is_valid_index = 0
        for wall in walls_2d[:]:
            if wall["polygons_drywall"][0]["type"] == wall["polygons_drywall"][1]["type"] == "DISABLED":
                wall_line = [[wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']]]
                if not is_wall_valid[is_valid_index]:
                    walls_2d.remove(wall)
                    is_valid_index += 1
                    continue
                walls_null_room.append(wall_line)
                walls_null_id.append(wall["id"])
                is_valid_index += 1
        shapes = self.disconnected_shapes(walls_null_room, tolerance=10*(sum(scale)//2))
        shapes_isolated = list(filter(lambda shape: len(shape) <= 2, shapes))
        lines_isolated = [shape[0] for shape in shapes_isolated]
        shapes_null = list(filter(lambda shape: len(shape) > 2, shapes))
        polygonized, perimeter_lines_contours = list(), list()
        for shape in shapes_null:
            shape_bounding_box_x_min = min([min(line[0][0], line[0][2]) for line in shape])
            shape_bounding_box_x_max = max([max(line[0][0], line[0][2]) for line in shape])
            shape_bounding_box_y_min = min([min(line[0][1], line[0][3]) for line in shape])
            shape_bounding_box_y_max = max([max(line[0][1], line[0][3]) for line in shape])
            lines_isolated_included = list()
            for line_isolated in lines_isolated:
                X1, Y1, X2, Y2 = line_isolated[0]
                X1_in_bound = (X1 >= shape_bounding_box_x_min - 20 and X1 <= shape_bounding_box_x_max + 20)
                X2_in_bound = (X2 >= shape_bounding_box_x_min - 20 and X2 <= shape_bounding_box_x_max + 20)
                Y1_in_bound = (Y1 >= shape_bounding_box_y_min - 20 and Y1 <= shape_bounding_box_y_max + 20)
                Y2_in_bound = (Y2 >= shape_bounding_box_y_min - 20 and Y2 <= shape_bounding_box_y_max + 20)
                if X1_in_bound and X2_in_bound and Y1_in_bound and Y2_in_bound:
                    lines_isolated_included.append(line_isolated)
            polygon_vertices = load_polygon(shape, lines_isolated_included)
            if not is_valid(polygon_vertices):
                continue
            perimeter_lines_contour_all = self.load_perimeter(polygon_vertices, [[[wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']]] for wall in walls_2d], scale=scale)
            polygon_area = cv2.contourArea(np.array(polygon_vertices, np.int32)) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["area"]
            for wall_line in perimeter_lines_contour_all:
                wall_payload = load_wall_payload(wall_line)
                if wall_line in shape:
                    walls_2d.remove(wall_payload)
                    continue
                if wall_line in lines_isolated_included:
                    walls_2d.remove(wall_payload)
                    continue
                polygon_payload = wall_payload["polygons_drywall"][0] if not wall_payload["polygons_drywall"][0]["enabled"] else wall_payload["polygons_drywall"][1]
                wall_payload["polygons_drywall"].remove(polygon_payload)

            perimeter_lines_contours.append(perimeter_lines_contour_all)
            polygonized.append((polygon_area, polygon_vertices))

        return polygonized, perimeter_lines_contours, walls_2d

    def save_plot_2d(
        self,
        model_2d_path,
        floor_plan_path="/tmp/floor_plan.png",
        overlay_enabled=False
    ):
        def put_text(canvas, text, origin, fontFace, fontScale, color, thickness, angle):
            text_image = np.zeros_like(canvas)
            cv2.putText(text_image, text, origin, fontFace, fontScale, color, thickness, cv2.LINE_AA)
            M = cv2.getRotationMatrix2D(origin, angle, 1)
            rotated_text_image = cv2.warpAffine(text_image, M, (canvas.shape[1], canvas.shape[0]))
            canvas_text_added = cv2.add(canvas, rotated_text_image)
            return canvas_text_added

        with open(model_2d_path, 'r') as f:
            walls_2d_with_polygons = json.load(f)
        walls_2d, polygons = walls_2d_with_polygons

        if overlay_enabled:
            canvas = cv2.imread(floor_plan_path)
        else:
            floor_plan_image = cv2.imread(floor_plan_path)
            height, width, _ = floor_plan_image.shape
            canvas = np.ones((height, width), dtype=np.uint8) * 255
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for polygon in polygons:
            canvas_to_overlay = canvas.copy()
            vertices = np.array(polygon["vertices"]).reshape((-1, 1, 2))
            color = tuple(polygon["polygon_drywall"]["color"])
            cv2.fillPoly(canvas_to_overlay, pts=[vertices], color=color)
            canvas = cv2.addWeighted(canvas_to_overlay, 0.3, canvas, 0.7, 0)

        for wall in walls_2d:
            self._draw_line(wall["wall_line"], canvas, overlay_enabled=overlay_enabled)
            for drywall in wall["polygons_drywall"]:
                pts = np.array([
                    [drywall["polygon"][0]['x'], drywall["polygon"][0]['y']],
                    [drywall["polygon"][1]['x'], drywall["polygon"][1]['y']],
                    [drywall["polygon"][2]['x'], drywall["polygon"][2]['y']],
                    [drywall["polygon"][3]['x'], drywall["polygon"][3]['y']]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                if drywall["enabled"]:
                    canvas = cv2.fillPoly(canvas, pts=[pts], color=drywall["color"])
                else:
                    canvas = cv2.fillPoly(canvas, pts=[pts], color=(0, 0, 255))
                if overlay_enabled:
                    canvas_annotation_origin_X = int(round((drywall["polygon"][0]['x'] + drywall["polygon"][1]['x'] + drywall["polygon"][2]['x'] + drywall["polygon"][3]['x']) / 4))
                    canvas_annotation_origin_Y = int(round((drywall["polygon"][0]['y'] + drywall["polygon"][1]['y'] + drywall["polygon"][2]['y'] + drywall["polygon"][3]['y']) / 4))
                    if drywall["enabled"]:
                        drywall_type = drywall["type"]
                    else:
                        drywall_type = "NULL"
                    orientation = self.classify_line(
                        wall["wall_line"][0]['x'],
                        wall["wall_line"][0]['y'],
                        wall["wall_line"][1]['x'],
                        wall["wall_line"][1]['y']
                    )
                    if orientation == "horizontal":
                        angle = 0
                    if orientation == "vertical":
                        angle = 90
                    if orientation == "inclined":
                        delta_x = wall["wall_line"][1]['x'] - wall["wall_line"][0]['x']
                        delta_y = wall["wall_line"][1]['y'] - wall["wall_line"][0]['y']
                        angle_radians = math.atan2(delta_y, delta_x)
                        angle = math.degrees(angle_radians)
                    canvas = put_text(
                        canvas,
                        drywall_type,
                        (canvas_annotation_origin_X, canvas_annotation_origin_Y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        angle
                    )

        if overlay_enabled:
            image_path = "/tmp/blueprint_model_2d_overlay_enabled.png"
        else:
            image_path = "/tmp/blueprint_model_2d.png"
        cv2.imwrite(image_path, canvas)
        return Path(image_path)

    def _load_nearest_transcription_blocks(
        self,
        polygon_centroid,
        transcription_block_with_centroids,
        threshold=1000
    ):
        nearest_neighbors = dict()
        X, Y = polygon_centroid
        for transcription, centroid in transcription_block_with_centroids.items():
            X_target, Y_target = centroid
            if math.hypot(X - X_target, Y - Y_target) <= threshold:
                nearest_neighbors[transcription] = centroid

        return nearest_neighbors

    def model(
        self,
        image_path="/tmp/floor_plan_wall_segmented.png",
        model_2d_path="/tmp/walls_2d.json",
        floor_plan_path="/tmp/floor_plan.png",
        output_path="/tmp/blueprint_model_2d.png",
        transcription_block_with_centroids=dict(),
        transcription_headers_and_footers=dict(),
    ):
        image_GRAY = self.read_floor_plan(image_path)
        output_path = Path(output_path)

        canvas = cv2.imread(floor_plan_path)
        height, width, _ = canvas.shape
        scale_x = width / 1920
        scale_y = height / 1080
        height_default = self._load_ceiling_height_and_scale(
            [
                canvas.copy()[-round(height / 3):, :],
                canvas.copy()[:, -round(width / 3):],
            ]
        )["ceiling_height"]
        wall_lines = self._patch_to_line(image_GRAY, floor_plan_path, (scale_x, scale_y))
        if not wall_lines:
            return None, None, None, None
        polygons, polygons_perimeter_walls, external_contour = self.polygonize(wall_lines)
        if len(polygons) < 5:
            return None, None, None, None
        external_contour_normalized = [(round(scale_x * coordinate[0]), round(scale_y * coordinate[1])) for coordinate in external_contour]
        perimeter_lines, outer_drywall_surfaces = self.perimeter_lines(wall_lines)
        polygon_vertices_normalized_all = list()
        futures = list()
        with ThreadPoolExecutor(max_workers=8) as executor:
            for index, ((polygon_area, polygon_vertices), polygon_perimeter_walls) in enumerate(zip(polygons, polygons_perimeter_walls)):
                index += 1
                polygon_vertices_normalized = [(round(scale_x * vertex[0]), round(scale_y * vertex[1])) for vertex in polygon_vertices]
                polygon_vertices_normalized_all.append(polygon_vertices_normalized)
                polygon_perimeter_walls_normalized = list()
                for polygon_perimeter_wall in polygon_perimeter_walls:
                    X1, Y1, X2, Y2 = polygon_perimeter_wall[0]
                    polygon_perimeter_wall_normalized = [[round(scale_x * X1), round(scale_y * Y1), round(scale_x * X2), round(scale_y * Y2)]]
                    polygon_perimeter_walls_normalized.append(polygon_perimeter_wall_normalized)
                polygon_area_normalized = polygon_area * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["area"]
                drywall_polygons = self._extrude_polygon_drywalls(
                    polygon_perimeter_walls_normalized,
                    polygon_vertices_normalized,
                    (scale_x, scale_y)
                )
                futures.append(executor.submit(
                    self._add_walls_polygon,
                    polygon_vertices_normalized,
                    polygon_area_normalized,
                    polygon_perimeter_walls_normalized,
                    drywall_polygons,
                    (scale_x, scale_y),
                    height_default,
                    floor_plan_path,
                    transcription_block_with_centroids,
                    index,
                ))
            [future.result() for future in futures]
            futures = list()
            for perimeter_line, outer_drywall_surface in zip(perimeter_lines, outer_drywall_surfaces):
                perimeter_polygons = self._extrude_polygon_perimeter(
                    perimeter_line,
                    (scale_x, scale_y),
                    outer_drywall_surface=outer_drywall_surface
                )
                futures.append(executor.submit(
                    self._add_wall_perimeter,
                    perimeter_line,
                    perimeter_polygons,
                    height_default,
                    (scale_x, scale_y),
                ))
            [future.result() for future in futures]

        self._walls_2d = self._normalize_walls_2d(self._walls_2d, (scale_x, scale_y))
        missing_polygons, missing_polygons_perimeter_walls, self._walls_2d = self._load_missing_polygons(
            self._walls_2d,
            (scale_x, scale_y),
            polygon_vertices_normalized_all,
            floor_plan_path
        )
        external_contour_normalized = self.merge_polygons(external_contour_normalized, [polygon[1] for polygon in missing_polygons])
        futures = list()
        with ThreadPoolExecutor(max_workers=8) as executor:
            for index, ((polygon_area, polygon_vertices), polygon_perimeter_walls) in enumerate(zip(missing_polygons, missing_polygons_perimeter_walls)):
                index += len(polygons)
                drywall_polygons = self._extrude_polygon_drywalls(polygon_perimeter_walls, polygon_vertices, (scale_x, scale_y))
                futures.append(executor.submit(
                    self._add_walls_polygon,
                    polygon_vertices,
                    polygon_area,
                    polygon_perimeter_walls,
                    drywall_polygons,
                    (scale_x, scale_y),
                    height_default,
                    floor_plan_path,
                    transcription_block_with_centroids,
                    index,
                ))
            [future.result() for future in futures]
        self._walls_2d = self._normalize_walls_2d(self._walls_2d, (scale_x, scale_y))
        self._polygons = self._normalize_polygons(self._polygons, self._walls_2d)
        if model_2d_path:
            with open(model_2d_path, 'w') as f:
                json.dump([self._walls_2d, self._polygons], f, indent=2)
        return self._walls_2d, self._polygons, model_2d_path, external_contour_normalized
