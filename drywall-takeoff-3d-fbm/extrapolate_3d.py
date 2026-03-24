from copy import deepcopy
import json
from pathlib import Path

from pdf2image import convert_from_path
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from floor_plan import FloorPlan
from gltf_generator import load_gltf

__all__ = ["Extrapolate3D"]


class Extrapolate3D(FloorPlan):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self._hyperparameters = hyperparameters["modelling"]
        self._height_in_feet = self._hyperparameters["height_in_feet"]
        self._height_in_pixels = int(round(self._height_in_feet / min(self._hyperparameters["pixel_aspect_ratio"]["vertical"], self._hyperparameters["pixel_aspect_ratio"]["horizontal"])))
        self._walls_3d = list()
        self._polygons_3d = list()

    def _load_wall_width_in_pixels(self, wall_line, half=True):
        X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
        orientation = self.classify_line(X1, Y1, X2, Y2)
        wall_width = wall_line["thickness"]
        if orientation == "horizontal":
            if half:
                return int(round(wall_width / self._hyperparameters["pixel_aspect_ratio"]["horizontal"])) / 2
            return int(round(wall_width / self._hyperparameters["pixel_aspect_ratio"]["horizontal"]))
        if orientation == "vertical":
            if half:
                return int(round(wall_width / self._hyperparameters["pixel_aspect_ratio"]["vertical"])) / 2
            return int(round(wall_width / self._hyperparameters["pixel_aspect_ratio"]["vertical"]))
        if half:
            return math.hypot(
                int(round(wall_width / self._hyperparameters["pixel_aspect_ratio"]["horizontal"])),
                int(round(wall_width / self._hyperparameters["pixel_aspect_ratio"]["vertical"]))
            ) / 2
        return math.hypot(
            int(round(wall_width / self._hyperparameters["pixel_aspect_ratio"]["horizontal"])),
            int(round(wall_width / self._hyperparameters["pixel_aspect_ratio"]["vertical"]))
        )

    def _load_wall_height_in_pixels(self, wall_line):
        wall_height = wall_line["height"]
        if not wall_height:
            return self._height_in_pixels
        pixel_aspect_ratio_average = (self._hyperparameters["pixel_aspect_ratio"]["horizontal"] + self._hyperparameters["pixel_aspect_ratio"]["vertical"]) / 2
        return round(wall_height / pixel_aspect_ratio_average)

    def _load_model_2d(self, model_2d_path):
        with open(model_2d_path, 'r') as f:
            lines = json.load(f)
        return lines

    def _load_polygons(self, polygons_path):
        with open(polygons_path, 'r') as f:
            polygons = json.load(f)
        return polygons

    def _extrude_height_polygon(self, height_in_pixels, polygon):
        height_extruded = list()
        for line in polygon:
            line_bottom = list()
            for coordinate in deepcopy(line):
                coordinate['x'] = int(coordinate['x'])
                coordinate['y'] = int(coordinate['y'])
                coordinate['z'] = 0
                line_bottom.append(coordinate)
            line_top = list()
            for coordinate in deepcopy(line[::-1]):
                coordinate['x'] = int(coordinate['x'])
                coordinate['y'] = int(coordinate['y'])
                coordinate['z'] = height_in_pixels
                line_top.append(coordinate)
            height_extruded.append(line_bottom+line_top)
        return height_extruded

    def _extrude_width_with_arbritrary_orientation(self, x1, y1, x2, y2, width):
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5

        nx = -dy / length
        ny =  dx / length

        half = self._load_wall_width_in_pixels(dict(wall_line=[dict(x=x1, y=y1), dict(x=x2, y=y2)], thickness=width))
        ox = nx * half
        oy = ny * half

        front_face = [
            { 'x': x1 + ox, 'y': y1 + oy},
            { 'x': x2 + ox, 'y': y2 + oy}
        ]
        back_face = [
            { 'x': x2 - ox, 'y': y2 - oy},
            { 'x': x1 - ox, 'y': y1 - oy}
        ]
        return front_face, back_face

    def _extrude_width(self, wall_line):
        X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
        orientation = self.classify_line(X1, Y1, X2, Y2)
        width_half = self._load_wall_width_in_pixels(wall_line)
        if orientation == "horizontal":
            front_face = [
                { 'x': X1, 'y': max(0, Y1 - width_half)},
                { 'x': X2, 'y': max(0, Y2 - width_half)}
            ]
            back_face = [
                { 'x': X1, 'y': Y1 + width_half},
                { 'x': X2, 'y': Y2 + width_half}
            ]
            return front_face, back_face
        if orientation == "vertical":
            front_face = [
                { 'x': max(0, X1 - width_half), 'y': Y1},
                { 'x': max(0, X2 - width_half), 'y': Y2}
            ]
            back_face = [
                { 'x': X1 + width_half, 'y': Y1},
                { 'x': X2 + width_half, 'y': Y2}
            ]
            return front_face, back_face
        if orientation == "inclined":
            return self._extrude_width_with_arbritrary_orientation(X1, Y1, X2, Y2, wall_line["thickness"])
        return None, None

    def _is_mitered_butt(self, wall_line, wall_line_orientation, horizontal_wall_lines, vertical_wall_lines):
        X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
        if wall_line_orientation == "horizontal":
            target_neighbors = vertical_wall_lines
        if wall_line_orientation == "vertical":
            target_neighbors = horizontal_wall_lines
        is_mitered_butt = dict(A=set(), B=set())
        for target_wall_line in target_neighbors:
            target_X1, target_Y1, target_X2, target_Y2 = target_wall_line["x1"], target_wall_line["y1"], target_wall_line["x2"], target_wall_line["y2"]
            if min(math.hypot((X1 - target_X1), (Y1 - target_Y1)), math.hypot((X1 - target_X2), (Y1 - target_Y2))) <= self._hyperparameters["tolerance_euclidean_join"]:
                if wall_line_orientation == "horizontal" and math.hypot((X1 - target_X1), (Y1 - target_Y1)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (Y1 - target_Y1) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['A'].add("orientation_A")
                    if 0 <= (target_Y1 - Y1) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['A'].add("orientation_B")
                if wall_line_orientation == "horizontal" and math.hypot((X1 - target_X2), (Y1 - target_Y2)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (Y1 - target_Y2) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['A'].add("orientation_A")
                    if 0 <= (target_Y2 - Y1) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['A'].add("orientation_B")
                if wall_line_orientation == "vertical" and math.hypot((X1 - target_X1), (Y1 - target_Y1)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (X1 - target_X1) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['A'].add("orientation_A")
                    if 0 <= (target_X1 - X1) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['A'].add("orientation_B")
                if wall_line_orientation == "vertical" and math.hypot((X1 - target_X2), (Y1 - target_Y2)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (X1 - target_X2) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['A'].add("orientation_A")
                    if 0 <= (target_X2 - X1) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['A'].add("orientation_B")
            if min(math.hypot((X2 - target_X1), (Y2 - target_Y1)), math.hypot((X2 - target_X2), (Y2 - target_Y2))) <= self._hyperparameters["tolerance_euclidean_join"]:
                if wall_line_orientation == "horizontal" and math.hypot((X2 - target_X1), (Y2 - target_Y1)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (Y2 - target_Y1) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['B'].add("orientation_A")
                    if 0 <= (target_Y1 - Y2) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['B'].add("orientation_B")
                if wall_line_orientation == "horizontal" and math.hypot((X2 - target_X2), (Y2 - target_Y2)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (Y2 - target_Y2) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['B'].add("orientation_A")
                    if 0 <= (target_Y2 - Y2) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['B'].add("orientation_B")
                if wall_line_orientation == "vertical" and math.hypot((X2 - target_X1), (Y2 - target_Y1)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (X2 - target_X1) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['B'].add("orientation_A")
                    if 0 <= (target_X1 - X2) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['B'].add("orientation_B")
                if wall_line_orientation == "vertical" and math.hypot((X2 - target_X2), (Y2 - target_Y2)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (X2 - target_X2) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['B'].add("orientation_A")
                    if 0 <= (target_X2 - X2) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['B'].add("orientation_B")

        return is_mitered_butt

    def _extrude_width_mitered_butt(self, wall_line, horizontal_wall_lines, vertical_wall_lines):
        X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
        orientation = self.classify_line(X1, Y1, X2, Y2)
        width_half = self._load_wall_width_in_pixels(wall_line)
        if orientation == "horizontal":
            mitered_butt = self._is_mitered_butt(wall_line, orientation, horizontal_wall_lines, vertical_wall_lines)
            if "orientation_A" in list(mitered_butt['A']) and "orientation_B" in list(mitered_butt['A']):
                front_face_A = {'x': X1, 'y': max(0, Y1 - width_half)}
                back_face_A = {'x': X1, 'y': Y1 + width_half}
            elif "orientation_A" in list(mitered_butt['A']):
                front_face_A = {'x': max(0, X1 - width_half), 'y': max(0, Y1 - width_half)}
                back_face_A = {'x': X1 + width_half, 'y': Y1 + width_half}
            elif "orientation_B" in list(mitered_butt['A']):
                front_face_A = {'x': X1 + width_half, 'y': max(0, Y1 - width_half)}
                back_face_A = {'x': max(0, X1 - width_half), 'y': Y1 + width_half}
            else:
                front_face_A = {'x': X1, 'y': max(0, Y1 - width_half)}
                back_face_A = {'x': X1, 'y': Y1 + width_half}

            if "orientation_A" in list(mitered_butt['B']) and "orientation_B" in list(mitered_butt['B']):
                front_face_B = {'x': X2, 'y': max(0, Y2 - width_half)}
                back_face_B = {'x': X2, 'y': Y2 + width_half}
            elif "orientation_A" in list(mitered_butt['B']):
                front_face_B = {'x': max(0, X2 - width_half), 'y': max(0, Y2 - width_half)}
                back_face_B = {'x': X2 + width_half, 'y': Y2 + width_half}
            elif "orientation_B" in list(mitered_butt['B']):
                front_face_B = {'x': X2 + width_half, 'y': max(0, Y2 - width_half)}
                back_face_B = {'x': max(0, X2 - width_half), 'y': Y2 + width_half}
            else:
                front_face_B = {'x': X2, 'y': max(0, Y2 - width_half)}
                back_face_B = {'x': X2, 'y': Y2 + width_half}

            front_face = [front_face_A, front_face_B]
            back_face = [back_face_A, back_face_B]
            return front_face, back_face
        if orientation == "vertical":
            mitered_butt = self._is_mitered_butt(wall_line, orientation, horizontal_wall_lines, vertical_wall_lines)
            if "orientation_A" in list(mitered_butt['A']) and "orientation_B" in list(mitered_butt['A']):
                front_face_A = {'x': max(0, X1 - width_half), 'y': Y1}
                back_face_A  = {'x': X1 + width_half, 'y': Y1}
            elif "orientation_A" in list(mitered_butt['A']):
                front_face_A = {'x': max(0, X1 - width_half), 'y': max(0, Y1 - width_half)}
                back_face_A  = {'x': X1 + width_half, 'y': Y1 + width_half}
            elif "orientation_B" in list(mitered_butt['A']):
                front_face_A = {'x': X1 + width_half, 'y': max(0, Y1 - width_half)}
                back_face_A  = {'x': max(0, X1 - width_half), 'y': Y1 + width_half}
            else:
                front_face_A = {'x': max(0, X1 - width_half), 'y': Y1}
                back_face_A  = {'x': X1 + width_half, 'y': Y1}

            if "orientation_A" in list(mitered_butt['B']) and "orientation_B" in list(mitered_butt['B']):
                front_face_B = {'x': max(0, X2 - width_half), 'y': Y2}
                back_face_B  = {'x': X2 + width_half, 'y': Y2}
            elif "orientation_A" in list(mitered_butt['B']):
                front_face_B = {'x': max(0, X2 - width_half), 'y': max(0, Y2 - width_half)}
                back_face_B  = {'x': X2 + width_half, 'y': Y2 + width_half}
            elif "orientation_B" in list(mitered_butt['B']):
                front_face_B = {'x': X2 + width_half, 'y': max(0, Y2 - width_half)}
                back_face_B  = {'x': max(0, X2 - width_half), 'y': Y2 + width_half}
            else:
                front_face_B = {'x': max(0, X2 - width_half), 'y': Y2}
                back_face_B  = {'x': X2 + width_half, 'y': Y2}

            front_face = [front_face_A, front_face_B]
            back_face = [back_face_A, back_face_B]
            return front_face, back_face
        if orientation == "inclined":
            return self._extrude_width_with_arbritrary_orientation(X1, Y1, X2, Y2, wall_line["thickness"])
        return None, None

    def _extrude_3d(self, wall_line, horizontal_wall_lines=list(), vertical_wall_lines=list()):
        if horizontal_wall_lines and vertical_wall_lines:
            front_face, back_face = self._extrude_width_mitered_butt(wall_line, horizontal_wall_lines, vertical_wall_lines)
        else:
            front_face, back_face = self._extrude_width(wall_line)
        if front_face and back_face:
            height_in_pixels = self._load_wall_height_in_pixels(wall_line)
            polygons = self._extrude_height_polygon(height_in_pixels, [front_face, back_face])
            return polygons

    def _add_wall(self, wall_line, polygons):
        wall = dict(
            id=wall_line["id"],
            thickness=wall_line["thickness"],
            height=wall_line["height"],
            length=wall_line["length"],
            type=wall_line["type"],
            surfaces_drywall=list(),
            wall_line=wall_line["wall_line"],
            drywall_choices=wall_line["drywall_choices"],
        )
        for polygon, polygon_type in zip(polygons, wall_line["polygons_drywall"]):
            wall["surfaces_drywall"].append(
                dict(
                    id=polygon_type["id"],
                    room_name=polygon_type["room_name"],
                    polygon=polygon,
                    type=polygon_type["type"],
                    type_stacked=polygon_type["type_stacked"],
                    enabled=polygon_type["enabled"],
                    thickness=polygon_type["thickness"],
                    layers=polygon_type["layers"],
                    fire_rating=polygon_type["fire_rating"],
                    recommendation=polygon_type["recommendation"],
                    color=polygon_type["color"],
                    color_stacked=polygon_type["color_stacked"],
                    waste_factor=polygon_type["waste_factor"],
                )
            )
        self._walls_3d.append(wall)

    def _extrude_roof_3d(self, vertices, slope, tilt_axis, height_in_pixels, width_in_pixels):
        if slope is None or slope == 0:
            height_front_face = height_in_pixels - (width_in_pixels // 2)
            height_back_face = height_in_pixels + (width_in_pixels // 2)
            front_face = [dict(x=vertex[0], y=vertex[1], z=height_front_face) for vertex in vertices]
            back_face = [dict(x=vertex[0], y=vertex[1], z=height_back_face) for vertex in vertices]
            return [front_face, back_face]

        xs = [vertex[0] for vertex in vertices]
        ys = [vertex[1] for vertex in vertices]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        front_face, back_face = list(), list()

        for x, y in vertices:
            if tilt_axis == "horizontal":
                d = cx - x
            elif tilt_axis == "vertical":
                d = cy - y
            else:
                slope = 0
                d = cx - x

            height_offset = math.tan(math.radians(slope)) * d
            height_front_face = height_in_pixels - height_offset
            height_back_face = height_in_pixels + height_offset

            front_face.append(dict(x=x, y=y, z=height_front_face))
            back_face.append(dict(x=x, y=y, z=height_back_face))
        return [front_face, back_face]

    def _add_polygon(self, polygon):
        height_in_pixels = self._load_wall_height_in_pixels(polygon)
        pixel_aspect_ratio_average = (self._hyperparameters["pixel_aspect_ratio"]["horizontal"] + self._hyperparameters["pixel_aspect_ratio"]["vertical"]) / 2
        width_in_pixels = round(polygon["polygon_drywall"]["thickness"] / pixel_aspect_ratio_average)
        polygon = dict(
            id=polygon["id"],
            area=polygon["area"],
            vertices=polygon["vertices"],
            type=polygon["type"],
            height=polygon["height"],
            slope=polygon["slope"],
            slope_enabled=polygon["slope_enabled"],
            tilt_axis=polygon["tilt_axis"],
            room_name=polygon["room_name"],
            surface_drywall_ids_interior=polygon["polygon_ids_drywall_interior"],
            drywall_choices=polygon["drywall_choices"],
            surface_drywall=dict(
                polygon=self._extrude_roof_3d(polygon["vertices"], polygon["slope"], polygon["tilt_axis"], height_in_pixels, width_in_pixels),
                type=polygon["polygon_drywall"]["type"],
                enabled=polygon["polygon_drywall"]["enabled"],
                layers=polygon["polygon_drywall"]["layers"],
                color=polygon["polygon_drywall"]["color"],
                waste_factor=polygon["polygon_drywall"]["waste_factor"],
            )
        )
        self._polygons_3d.append(polygon)

    def compute_updated_area_polygon(self, polygon_vertices, area, slope, tilt_axis):
        if slope is None or slope == 0:
            return area

        if tilt_axis == "horizontal":
            polygon_Ys = [vertex[1] for vertex in polygon_vertices]
            polygon_width = max(polygon_Ys) - min(polygon_Ys)
        if tilt_axis == "vertical":
            polygon_Xs = [vertex[0] for vertex in polygon_vertices]
            polygon_width = max(polygon_Xs) - min(polygon_Xs)
        a = slope / polygon_width
        return round(area * math.sqrt(1 + a * a), 2)

    def save_plot_3d(self, model_3d_path, polygons_3d_path):
        def add_side_face(ax, p1_i, p2_i, p1_o, p2_o):
            face = [
                (p1_i["x"], 1080 - p1_i["y"], p1_i["z"]),
                (p2_i["x"], 1080 - p2_i["y"], p2_i["z"]),
                (p2_o["x"], 1080 - p2_o["y"], p2_o["z"]),
                (p1_o["x"], 1080 - p1_o["y"], p1_o["z"]),
            ]

            coll = Poly3DCollection([face], alpha=0.5)
            coll.set_edgecolor('k')
            ax.add_collection3d(coll)

        def add_roof_face(ax, vertices, height, color=(0.6, 0.6, 0.6)):
            verts_3d = [
                (x, 1080 - y, height)
                for x, y in vertices
            ]

            coll = Poly3DCollection([verts_3d], alpha=0.3)
            coll.set_facecolor(color)
            coll.set_edgecolor("k")
            ax.add_collection3d(coll)

        with open(model_3d_path, 'r') as f:
            walls_3d = json.load(f)

        xs, ys, zs = list(), list(), list()

        dpi = 100
        fig = plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        for wall in walls_3d:
            surfaces = [s for s in wall["surfaces_drywall"]]

            if len(surfaces) != 2:
                continue

            surf_A = surfaces[0]["polygon"]
            surf_B = surfaces[1]["polygon"]

            if len(surf_A) != 4 or len(surf_B) != 4:
                continue
    
            for i in range(4):
                p1_A = surf_A[i]
                p2_A = surf_A[(i + 1) % 4]

                p1_B = surf_B[i]
                p2_B = surf_B[(i + 1) % 4]

                add_side_face(ax, p1_A, p2_A, p1_B, p2_B)

            for surf in surfaces:
                poly = surf["polygon"]
                verts = [(p["x"], 1080 - p["y"], p["z"]) for p in poly]

                poly3d = [verts]
                coll = Poly3DCollection(poly3d, alpha=0.5)
                coll.set_edgecolor('k')
                ax.add_collection3d(coll)

        with open(polygons_3d_path, 'r') as f:
            polygons_3d = json.load(f)

        for polygon in polygons_3d:
            vertices = polygon["vertices"]
            height = self._load_wall_height_in_pixels(polygon)
            color = polygon["surface_drywall"]["color"][::-1]
            color = tuple(c / 255 for c in color)

            add_roof_face(
                ax,
                vertices=vertices,
                height=height,
                color=color
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if xs and ys and zs:
            ax.set_xlim(0, 1920)
            ax.set_ylim(1080, 0)
            ax.set_zlim(min(zs), max(zs))

        ax.view_init(elev=90, azim=-90)

        plt.tight_layout()
        image_path = "/tmp/blueprint_model_3d.png"
        plt.savefig(image_path, dpi=dpi)
        return Path(image_path)

    def gltf(self, model_2d_path="/tmp/walls_2d.json", polygons_path="/tmp/polygons.json"):
        wall_lines = self._load_model_2d(model_2d_path)
        polygons = self._load_polygons(polygons_path)
        walls = list()
        for wall_line in wall_lines:
            walls.append(
                dict(
                    x1=wall_line["wall_line"][0]['x'], 
                    y1=(1080 - wall_line["wall_line"][0]['y']), 
                    x2=wall_line["wall_line"][1]['x'], 
                    y2=(1080 - wall_line["wall_line"][1]['y']), 
                    height=wall_line["height"], 
                    thickness=wall_line["thickness"]
                )
            )
        load_gltf(walls, polygons, "/tmp/walls.gltf")
        return [Path("/tmp/walls.gltf"), Path("/tmp/walls.bin")]

    def extrapolate_wall_heights_given_polygons(self, walls_3d, polygons):
        def load_payload_wall_3d(wall_line):
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line = [dict(x=X1, y=Y1), dict(x=X2, y=Y2)]
            for wall in walls_3d:
                if wall["wall_line"] == wall_line:
                    return wall

        wall_lines = list()
        for wall_3d in walls_3d:
            wall_lines.append([[wall_3d["wall_line"][0]['x'], wall_3d["wall_line"][0]['y'], wall_3d["wall_line"][1]['x'], wall_3d["wall_line"][1]['y']]])
        for polygon in polygons:
            slope = polygon["slope"]
            if slope is None or slope == 0:
                continue
            tilt_axis = polygon["tilt_axis"]
            perimeter_lines_contour = self.load_perimeter(polygon["vertices"], wall_lines)
            for perimeter_line in perimeter_lines_contour:
                X1, Y1, X2, Y2 = perimeter_line[0]
                orientation = self.classify_line(X1, Y1, X2, Y2)
                payload = load_payload_wall_3d(perimeter_line)
                if tilt_axis == "horizontal":
                    a = polygon["slope"] / (max([vertex[1] for vertex in polygon["vertices"]]) - min([vertex[1] for vertex in polygon["vertices"]]))
                    if polygon["slope"] > 0:
                        roof_high_Y = min([vertex[1] for vertex in polygon["vertices"]])
                    else:
                        roof_high_Y = max([vertex[1] for vertex in polygon["vertices"]])
                    if orientation == "horizontal":
                        payload["height"] = round(max(0, polygon["height"] - a * (round(np.median([Y1, Y2])) - roof_high_Y)))
                    if orientation == "vertical":
                        payload["height"] = polygon["height"]
                elif tilt_axis == "vertical":
                    a = polygon["slope"] / (max([vertex[0] for vertex in polygon["vertices"]]) - min([vertex[0] for vertex in polygon["vertices"]]))
                    if polygon["slope"] > 0:
                        roof_high_X = min([vertex[0] for vertex in polygon["vertices"]])
                    else:
                        roof_high_X = max([vertex[0] for vertex in polygon["vertices"]])
                    if orientation == "vertical":
                        payload["height"] = round(max(0, polygon["height"] - a * (round(np.median([X1, X2])) - roof_high_X)))
                    if orientation == "horizontal":
                        payload["height"] = polygon["height"]
                else:
                    continue
        return walls_3d, polygons

    def recompute_dimensions_walls_and_polygons(self, walls_3d_JSON, polygons_JSON, pixel_aspect_ratio_new, floor_plan_pdf_path):
        width, height = convert_from_path(floor_plan_pdf_path, dpi=400)[0].size
        scale_x = 1920 / width
        scale_y = 1080 / height
        walls_3d_JSON_updated, polygons_JSON_updated = list(), list()
        for wall_3d in walls_3d_JSON:
            X1, Y1, X2, Y2 = wall_3d["wall_line"][0]['x'], wall_3d["wall_line"][0]['y'], wall_3d["wall_line"][1]['x'], wall_3d["wall_line"][1]['y']
            X1_scaled, Y1_scaled, X2_scaled, Y2_scaled = scale_x * X1, scale_y * Y1, scale_x * X2, scale_y * Y2
            length_target = round(math.hypot(
                (X1_scaled - X2_scaled) * pixel_aspect_ratio_new["horizontal"],
                (Y1_scaled - Y2_scaled) * pixel_aspect_ratio_new["vertical"]
            ), 2)
            wall_3d["length"] = length_target
            walls_3d_JSON_updated.append(wall_3d)
        for polygon in polygons_JSON:
            polygon_vertices_scaled = [(scale_x * vertex[0], scale_y * vertex[1]) for vertex in polygon["vertices"]]
            polygon_vertices_scaled = np.array(polygon_vertices_scaled, np.float32)
            area = cv2.contourArea(polygon_vertices_scaled)
            polygon["area"] = pixel_aspect_ratio_new["area"] * area
            polygons_JSON_updated.append(polygon)

        return walls_3d_JSON_updated, polygons_JSON_updated

    def _scale_hyperparameters(self, scale):
        new_pixel_aspect_ratio_to_feet = self.compute_pixel_aspect_ratio(scale, self.hyperparameters["pixel_aspect_ratio_to_feet"])
        self.hyperparameters["pixel_aspect_ratio_to_feet"] = new_pixel_aspect_ratio_to_feet
        self.hyperparameters["modelling"]["pixel_aspect_ratio"] = new_pixel_aspect_ratio_to_feet
        self._hyperparameters = self.hyperparameters["modelling"]
        self._height_in_feet = self._hyperparameters["height_in_feet"]
        self._height_in_pixels = int(round(self._height_in_feet / min(self._hyperparameters["pixel_aspect_ratio"]["vertical"], self._hyperparameters["pixel_aspect_ratio"]["horizontal"])))

    def extrapolate(
        self,
        scale,
        model_2d_path="/tmp/walls_2d.json",
        polygons_path="/tmp/polygons.json",
        model_3d_path="/tmp/walls_3d.json",
        polygons_3d_path="/tmp/polygons_3d.json",
        mitered_butt_enabled=False
    ):
        self._scale_hyperparameters(scale)
        lines = self._load_model_2d(model_2d_path)
        polygons = self._load_polygons(polygons_path)
        horizontal_wall_lines, vertical_wall_lines = list(), list()
        if mitered_butt_enabled:
            for wall_line in lines:
                X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
                orientation = self.classify_line(X1, Y1, X2, Y2)
                if orientation == "horizontal":
                    horizontal_wall_lines.append(wall_line)
                if orientation == "vertical":
                    vertical_wall_lines.append(wall_line)
        for polygon in polygons:
            self._add_polygon(polygon)
        for line in lines:
            polygons = self._extrude_3d(
                line,
                horizontal_wall_lines=horizontal_wall_lines,
                vertical_wall_lines=vertical_wall_lines,
            )
            if polygons:
                self._add_wall(line, polygons)

        if model_3d_path:
            with open(model_3d_path, 'w') as f:
                json.dump(self._walls_3d, f, indent=2)
        if polygons_3d_path:
            with open(polygons_3d_path, 'w') as f:
                json.dump(self._polygons_3d, f, indent=2)
        return self._walls_3d, self._polygons_3d, model_3d_path, polygons_3d_path