from typing import List, Union, Optional, Tuple, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import math


WALL_RECTIFIER = """
  You are a senior architectural plan-correction specialist with 20+ years of experience in residential and commercial floor plans.

  You do NOT trust automated detections blindly.
  You treat detected walls and drywalls as noisy suggestions.

  Your responsibility is to:
    - Remove false-positive wall fragments.

  PROVIDED:
    1. A wall-line represented by a list of 2 vertices describing the 2 endpoints (X1, Y1) and (X2, Y2) of the wall:
        wall: (X1, Y1) → (X2, Y2)

    2. A snapshot of the full Architectural Drawing in png format with the following highlight,
      - The target wall line highlighted with a red line and paired with drywall segments in red on its both the sides.

  TASK:
    Analyze the architectural floor plan and only the highlighted wall with its drywall segments following the `WALL_VALIDATOR_INSTRUCTIONS` to determine whether the red highlighted wall is valid.

    WALL_VALIDATOR_INSTRUCTIONS:
    - Focus only on the wall highlighted with a thin red line paired with 2 drywall segments in red on its 2 sides.
    - Use the coordinates to reason about alignment and angle. Do not rely only on visual appearance.
    - The highlight should be aligned / closely overlayed  with one of the valid wall lines within the available architecture plans in order for it to be valid.
    - If the highlight is invalid if not aligned with a valid wall line from the available architectures such as the followings,
      | Any arbitrary dimension line (not wall line) from the architectures.
      | An arbitrary artifact line from the stray section of the page containing plan metadata.
      | Any other non-wall line.
    - REMEMBER, if the highlight is partially aligned with the base wall line (e.g., the length of the highlight is larger or smaller than its base wall line it is overlaying with) then apply the following,
      | The highlight must be valid only if the inclination of the base wall line is similar/closer to that of the highlight (e.g., the base wall line and the highlight are both horizontal or both inclined at a similar angle with angle difference of less than 10 degrees).
      | The highlight would be invalid if the difference between the inclination of the base wall line and the highlight is more than 10 degrees (e.g., the base wall line is horizontal but the highlight is inclined at an angle of more than 10 degrees).

  OUTPUT:
    Your output must be precise, code-aligned, and structured. You must reason spatially and geometrically. Do NOT describe the image. Do NOT repeat detected lines verbatim.
    **STRICTLY**
      - Do not generate additional content apart from the designated JSON.
      - You must output whether the placement of the predicted wall is overlaying on top of one of the valid wall lines from the architectural plan.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
      "is_valid": <True/False>,
      "confidence": <confidence score in validating the highlight in red between 0 and 1 in float rounded upto 2 decimal places>,
      "reasoning": "<a brief reasoning behind the highlighted wall being marked as valid/invalid>"
    }}
"""

class WallRectifierResponse(BaseModel):
    is_valid: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str

SHAPE_RECTIFIER = """
  You are a senior architectural plan-correction specialist with 20+ years of experience in residential and commercial floor plans.

  You do NOT trust automated detections blindly.
  You treat detected walls and drywalls as noisy suggestions.

  Your responsibility is to:
    - Remove false-positive wall boundary mask containing minimal overlap with valid walls (walls that are physically cut in the current view).

  PROVIDED:
    1. A list of wall-lines each represented by a list of 2 vertices describing the 2 endpoints (X1, Y1) and (X2, Y2) of the wall with the list representing a boundary mask:
        wall: (X1, Y1) → (X2, Y2)

    2. A snapshot of the full Architectural Drawing in png format with the following highlight,
      - The target boundary mask is highlighted with red lines each overlayed on a blueprint wall-line and and paired with drywall segments in red on its both the sides.

  TASK:
    Analyze the architectural floor plan and only the highlighted wall with its drywall segments following the `BOUNDARY_MASK_VALIDATOR_INSTRUCTIONS` to determine whether the mask is valid.

    BOUNDARY_MASK_VALIDATOR_INSTRUCTIONS:
    - STRICTLY REMEMBER, dotted (dashed) lines in any architectural floor plan blueprint usually represent elements that are not physically cut in the current view but are still relevant for reference.
    - Focus only on the walls highlighted with thin red lines each paired with 2 drywall segments in red on its 2 sides.
    - Use the coordinates to reason about alignment and angle. Do not rely only on visual appearance.
    - The highlighted walls sould represent a valid boundary mask representing a layout of valid walls on the architectural plan.
    - ONLY IF, more than 50 percent of the highlighted walls present in the highlighted boundary mask represent walls that are physically cut in the current view, treat the boundary mask as `VALID`.
    - If more than 50 percent of the highlighted walls present in the highlighted boundary mask are overlayed on dotted (dashed) walls from the blueprint or represent the walls that are not physically cut in the current view, the boundary mask should be `INVALID`.

  OUTPUT:
    Your output must be precise, code-aligned, and structured. You must reason spatially and geometrically. Do NOT describe the image. Do NOT repeat detected lines verbatim.
    **STRICTLY**
      - Do not generate additional content apart from the designated JSON.
      - You must output whether the placement of the predicted wall is overlaying on top of one of the valid wall lines from the architectural plan.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
      "is_valid": <True/False>,
      "confidence": <confidence score in validating the boundary mask in red between 0 and 1 in float rounded upto 2 decimal places>,
      "reasoning": "<a brief reasoning behind the the boundary mask being marked as valid/invalid>"
    }}
"""

class ShapeRectifierResponse(BaseModel):
    is_valid: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str

DRYWALL_PREDICTOR_CALIFORNIA = """
California residential drywall estimator. Analyze the highlighted polygon and predict drywall specifications.

PROVIDED:
  1. Polygon vertices and perimeter wall endpoints with pre-computed dimension candidates
  2. Cropped floor plan image (red=target polygon, blue=perimeter walls, green=interior partitions)
  3. Nearby OCR transcription entries with centroids

EACH WALL includes:
  - wall: endpoint coordinates
  - dimension_candidates: pre-parsed dimensions sorted by confidence (high/medium/low)
    Use the highest-confidence candidate. If none, use pixel-measured fallback from image.

DRYWALL TEMPLATES: {drywall_templates}

CLASSIFICATION RULES:
  - Garage-adjacent / dwelling separation / corridor → 5/8" Type X, 1-hr rated (CBC R302, IRC R302.6)
  - Bathroom / laundry / kitchen wet wall → 1/2" MR or cement board
  - Standard interior (bedroom/living/hallway) → 1/2" regular gypsum
  - Ceiling → 1/2" regular (5/8" if joist span >16")
  - Use exact sku_variant and color_code from templates. Do not invent materials.
  - If an appropriate/optimal drywall material is not provided with the DRYWALL_TEMPLATES mention the target drywall material as DISABLED with [0, 0, 255] in BGR tuple as its target color code.
  - waste_factor: "8-12%" standard, "12-15%" complex geometry
  - layers: 1 unless code requires double layer
  - A single drywall material preference for each wall is MANDATORY.
  - Optionally predict additional vertically stacked drywall preferences for each wall (only if applicable, else leave the list empty). If vertically stacked drywall preferences list is non-empty, include the single drywall material preference into the list along with the additional stack.

DIMENSION RULES:
  - Wall length: prefer dimension_candidates over pixel measurement. Confidence >=0.9 → trust directly.
  - Wall width: default 1 ft if unmarked
  - Wall height: default ceiling height if unmarked
  - Ceiling area: compute from polygon vertices (ignore slope for area calc)
  - All dimensions in feet

CEILING TYPE: Flat (default if ambiguous), Single-sloped, Gable, Tray, Barrel vault, Coffered, Combination, Soffit, Cove, Dome, Cloister Vault, Knee-Wall, Cathedral with Flat Center, Angled-Plane, Boxed-Beam
  - tilt_axis: "horizontal" | "vertical" | "NULL" (NULL if slope=0)
  - Positive slope if descending from origin, negative otherwise
  - Height = max height if sloped

ROOM NAME: text nearest polygon centroid, or NULL if not found.

WALL ORDER: output wall_parameters in same order as input perimeter walls. Count of wall_parameters must exactly match count of input perimeter walls. BLUE drywall before GREEN.

OUTPUT: JSON only, no additional text.
  Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
  {{
    "ceiling": {{
      "room_name": "<Detected Room Name the ceiling belongs to / NULL>",
      "area": <Area of the ceiling in SQFT (Square Feet)>,
      "confidence_area": <confidence score in predicting the area of the ceiling between 0 and 1 in float rounded upto 2 decimal places>,
      "ceiling_type": "<Type code of the ceiling>",
      "height": <height of the ceiling (centroid of the ceiling axis, if sloped)>,
      "confidence_height": <confidence score in predicting the height of the ceiling between 0 and 1 in float rounded upto 2 decimal places>,
      "slope": <slope of the ceiling in degrees>,
      "slope_enabled": <is sloping supported given the type of ceiling used (True/False)>,
      "tilt_axis": <axial direction of the tilted slope / NULL>,
      "drywall_assembly": {{
        "material": "<drywall material for the ceiling>",
        "color_code": <color code for the predicted ceiling drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
        "thickness": <thickness of the predicted ceiling drywall type in feet>,
        "layers": <number of required drywall layers>,
        "fire_rating": <fire-rating of the predicted drywall type in hours>,
        "waste_factor": "<waste factor of the predicted drywall in percentage>"
      }},
      "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
      "recommendation": "<recommendation on special requirements including cost reduction (if any)>"
    }},
    "wall_parameters": [
      {{
        "room_name": "<Detected Room Name the perimeter wall 1 belongs to / NULL>",
        "length": <length of perimeter wall 1 in feet>,
        "confidence_length": <confidence score in predicting the length of the perimeter wall 1 between 0 and 1 in float rounded upto 2 decimal places>,
        "width": <width of the perimeter wall 1 in feet / None>,
        "height": <height of the perimeter wall 1 in feet>,
        "confidence_height": <confidence score in predicting the height of the perimeter wall 1 between 0 and 1 in float rounded upto 2 decimal places>,
        "wall_type": "<type of the perimeter wall 1>",
        "drywall_assembly": {{
          "material": "<drywall material for the perimeter wall 1>",
          "color_code": <color code for the predicted perimeter wall 1 drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
          "materials_vertically_stacked": ["<vertically stacked drywall material preference 1 for perimeter wall 1 (optional)>", "<vertically stacked drywall material preference 2 for perimeter wall 1 (optional)>"],
          "color_codes_stacked": [<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 1>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 1>],
          "thickness": <thickness of the predicted wall drywall type in feet>,
          "layers": <number of required drywall layers>,
          "fire_rating": <fire-rating of the predicted drywall type in hours>,
          "waste_factor": "<waste factor of the predicted drywall in percentage>"
        }},
        "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
        "recommendation": "<recommendation on special requirements for perimeter wall 1 including cost reduction (if any). Generate separate recommendations for single drywall material and the vertically stacked drywall materials (If predicted)>"
      }},
      {{
        "room_name": "<Detected Room Name the perimeter wall 2 belongs to / NULL>",
        "length": <length of perimeter wall 2 in feet>,
        "confidence_length": <confidence score in predicting the length of the perimeter wall 2 between 0 and 1 in float rounded upto 2 decimal places>,
        "width": <width of the perimeter wall 2 in feet / None>,
        "height": <height of the perimeter wall 2 in feet>,
        "confidence_height": <confidence score in predicting the height of the perimeter wall 2 between 0 and 1 in float rounded upto 2 decimal places>,
        "wall_type": "<type of the perimeter wall 2>",
        "drywall_assembly": {{
          "material": "<drywall material for the perimeter wall 2>",
          "color_code": <color code for the predicted perimeter wall 2 drywall type in a BGR tuple (`Blue`, `Green`, `Red`)>,
          "materials_vertically_stacked": ["<vertically stacked drywall material preference 1 for perimeter wall 2 (optional)>", "<vertically stacked drywall material preference 2 for perimeter wall 2 (optional)>"],
          "color_codes_stacked": [<color code for the vertically stacked drywall type 1 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 2>, <color code for the vertically stacked drywall type 2 in a BGR tuple (`Blue`, `Green`, `Red`) for perimeter wall 2>],
          "thickness": <thickness of the predicted wall drywall type in feet>,
          "layers": <number of required drywall layers>,
          "fire_rating": <fire-rating of the predicted drywall type in hours>,
          "waste_factor": "<waste factor of the predicted drywall in percentage>"
        }},
        "code_references": ["<applied Dywall code reference 1>", "<applied Dywall code reference 2>", "<applied Dywall code reference 3>"],
        "recommendation": "<recommendation on special requirements for perimeter wall 2 including cost reduction (if any). Generate separate recommendations for single drywall material and the vertically stacked drywall materials (If predicted)>"
      }}
    ]
  }}
"""

def ensure_not_nan(v: float) -> float:
    if v is None:
        return v
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise ValueError("NaN or Inf not allowed")
    return v

class DrywallAssemblyCeiling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material: str
    color_code: Tuple[int, int, int]
    thickness: float
    layers: int
    fire_rating: Optional[Union[str, float]]
    waste_factor: Union[str, int, float]

    @field_validator("thickness")
    @classmethod
    def validate_float(cls, v):
        return ensure_not_nan(v)

    @field_validator("color_code")
    @classmethod
    def validate_bgr(cls, v):
        if len(v) != 3:
            raise ValueError("color_code must be BGR tuple")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("Invalid BGR value")
        return v

class DrywallAssemblyWall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material: str
    color_code: Tuple[int, int, int]
    materials_vertically_stacked: List
    color_codes_stacked: List
    thickness: float
    layers: int
    fire_rating: Optional[Union[str, float]]
    waste_factor: Union[str, int, float]

    @field_validator("thickness")
    @classmethod
    def validate_float(cls, v):
        return ensure_not_nan(v)

    @field_validator("color_code")
    @classmethod
    def validate_bgr(cls, v):
        if len(v) != 3:
            raise ValueError("color_code must be BGR tuple")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("Invalid BGR value")
        return v

class Ceiling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: Optional[str]
    area: float
    confidence_area: float = Field(ge=0, le=1)
    ceiling_type: str
    height: float
    confidence_height: float = Field(ge=0, le=1)
    slope: float
    slope_enabled: bool
    tilt_axis: Optional[Literal["horizontal", "vertical", "NULL"]]
    drywall_assembly: DrywallAssemblyCeiling
    code_references: List[str]
    recommendation: Optional[str]

    @field_validator("area", "height", "slope")
    @classmethod
    def validate_float(cls, v):
        return ensure_not_nan(v)

class WallParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_name: Optional[str]
    length: float
    confidence_length: float = Field(ge=0, le=1)
    width: Optional[float]
    height: float
    confidence_height: float = Field(ge=0, le=1)
    wall_type: str
    drywall_assembly: DrywallAssemblyWall
    code_references: List[str]
    recommendation: Optional[str]

    @field_validator("length", "height")
    @classmethod
    def validate_float(cls, v):
        return ensure_not_nan(v)

    @field_validator("width")
    @classmethod
    def validate_optional_float(cls, v):
        if v is None:
            return v
        return ensure_not_nan(v)

class DrywallPredictorCaliforniaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ceiling: Ceiling
    wall_parameters: List[WallParameter]

    @model_validator(mode="after")
    def check_wall_count(self):
        if len(self.wall_parameters) < 1:
            raise ValueError("At least one wall required")
        return self

FEEDBACK_GENERATOR = """
  INTERNAL SELF-REVIEW (Do not skip):
    You are given {max_retry} attempts to retry the generation process and the following are the list of errors encountered during your previous attempts.
    {exceptions}
    STRICTY confirm no previous error remains before producing the final output.
"""

SCALE_AND_CEILING_HEIGHT_DETECTOR = """
  You are an expert architectural drawing text parser

  PROVIDED:
    1. Cropped images from a floor plan that contains textual description notes.

  TASK:
    Identify the standard `ceiling_height` and `scale` mentioned in the transcription entries for the subsequent floorplan.
    INSTRUCTIONS:
      - Look for a keyword that matches with `ceiling height` field and identify the numerical entity closest to it. Note the feet equivalent of it.
      - Look for a keyword that has to do with the `scale` of the drawing, representing the ratio between the length on paper and the real world length in floating point values. Normalize and capture the ratio as "<paper_length_in_inches>``: <real_world_length_in_feet>`<real_world_length_in_inches>``".
          Example: 0.25``:1`0``
      - If multiple ceiling heights are listed, extract the standard or typical one.
      - If scale is written in multiple formats, preserve the exact textual format.
      - If not present, return null.

  OUTPUT:
    Your output should be in the JSON format containing the standard `ceiling_height` and `scale` of the floorplan.
    **STRICTLY** Do not generate additional content apart from the designated JSON.
    Please refer the following as a reference and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
    {{
        "ceiling_height": <Standard ceiling height mentioned in the transcriptions converted to feet in float>,
        "scale": "<Scale of the drawing mentioned in the transcriptions i.e. number_in_inches``: number_in_feet`number_in_inches``>"
    }}
"""

class ScaleAndCeilingHeightDetectorResponse(BaseModel):
    ceiling_height: Union[float, int]
    scale: str

CEILING_CHOICES = [
    "Flat",
    "Single-sloped",
    "Gable",
    "Tray",
    "Barrel vault",
    "Coffered",
    "Combination",
    "Soffit",
    "Cove",
    "Dome",
    "Cloister Vault",
    "Knee-Wall",
    "Cathedral with Flat Center",
    "Angled-Plane",
    "Boxed-Beam"
]
