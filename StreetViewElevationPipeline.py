import argparse
from collections.abc import Buffer
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


VEHICLE_HEIGHT_PRESETS_M = {"car": 1.5, "bus": 3.2, "truck": 3.8}
VEHICLE_LABELS = tuple(VEHICLE_HEIGHT_PRESETS_M.keys())
OBJECT_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
)


@dataclass
class VehicleDetection:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]
    assumed_height_m: float


@dataclass
class GroundAnchor:
    x: float
    y: float
    ground_height: float
    pixels_per_meter: float


@dataclass
class CalibrationCandidate:
    detection: VehicleDetection
    bbox_height_px: int
    expected_depth_m: float
    scale: float
    estimated_camera_height_m: float
    center_x: int
    ground_anchor_y: int
    top_depth_m: Optional[float]
    center_depth_m: float
    bottom_depth_m: float


@dataclass
class CalibrationSummary:
    enabled: bool
    applied: bool
    reason: str
    detections_found: int = 0
    detections_used: int = 0
    depth_scale_factor: Optional[float] = None
    camera_height_m: Optional[float] = None
    candidate_camera_height_m: Optional[float] = None
    local_ground_profile_created: bool = False
    local_ground_y_profile_created: bool = False
    ground_anchors: List[GroundAnchor] = field(default_factory=list)
    per_detection: List[CalibrationCandidate] = field(default_factory=list)

 
class ElevationMap:
    width: int = 0
    height: int = 0
    depth_map: Optional[np.ndarray] = None
    elevation_map: Optional[np.ndarray] = None
    elevation_valid_mask: Optional[np.ndarray] = None
    vehicle_detections: List[VehicleDetection] = []
    accepted_calibration_candidates: List[CalibrationCandidate] = []
    local_ground_profile: Optional[np.ndarray] = None
    local_ground_y_profile: Optional[np.ndarray] = None
    local_pixels_per_meter_profile: Optional[np.ndarray] = None
    calibration_summary = CalibrationSummary(
        enabled=True,
        applied=False,
        reason="not_run",
    )
    point_cloud = None

class CameraGeometry:
    def __init__(self, image: cv2.Mat, fov_degrees: float):
        self.height, self.width = image.shape[:2]
        fov_rad = np.radians(fov_degrees)
        self.fx = (self.width / 2.0) / np.tan(fov_rad / 2.0)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        self.x_norm = None
        self.y_norm = None
        self.vertical_angle = None
        self.sin_vertical_angle = None

        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        self.x_norm = (x_coords - self.cx) / self.fx
        self.y_norm = (y_coords - self.cy) / self.fy
        self.vertical_angle = np.arctan2(-self.y_norm, 1.0)
        self.sin_vertical_angle = np.sin(self.vertical_angle)

class DepthEstimator:
    """Placeholder for a depth estimation model loader."""
    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth from an image. To be implemented."""
        return None

class MiDaSDepthEstimator(DepthEstimator):
    
    def __init__(self):
        try:
            import torch
            from torchvision.transforms import Compose, Normalize, Resize, ToTensor

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS", trust_repo=True)
            except Exception:
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()

            self.transforms = Compose(
                [
                    Resize(384),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        except ImportError as exc:
            print(f"Failed to initialize MiDaS depth estimator: {exc}")
            self.model = None


    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        print("Using MiDas depth estimator...")
        if self.model is None:
            return None
        try:
            from PIL import Image

            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_tensor = self.transforms(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                depth_prediction = self.model(img_tensor)
                depth_map = torch.nn.functional.interpolate(
                    depth_prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_np = depth_map.cpu().numpy()
            depth_valid = depth_np[np.isfinite(depth_np)]
            if depth_valid.size == 0:
                return None

            depth_np_min = float(np.min(depth_valid))
            depth_np_max = float(np.max(depth_valid))
            if abs(depth_np_max - depth_np_min) < 1e-9:
                return None

            depth_norm = (depth_np - depth_np_min) / (depth_np_max - depth_np_min)
            return 0.5 + depth_norm * 29.5
        except Exception as exc:
            print(f"MiDaS depth estimation failed: {exc}")
            return None
    
class MediaPipeDepthEstimator(DepthEstimator):

    def _get_depth_model(self) -> str:
        """Return the MediaPipe depth model path if already installed."""
        model_path = Path.home() / ".cache" / "mediapipe" / "depth_model.tflite"
        if not model_path.exists():
            raise RuntimeError(
                "Download depth model from "
                "https://developers.google.com/mediapipe/solutions/vision/depth_estimator"
            )
        return str(model_path)

    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        print("Using MediaPipe depth estimator...")
        try:
            base_options = python.BaseOptions(model_asset_path=self._get_depth_model())
            options = vision.DepthEstimatorOptions(base_options=base_options)
            estimator = vision.DepthEstimator.create_from_options(options)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            depth_result = estimator.estimate(mp_image)
            depth_array = np.array(depth_result.depth_map)
            depth_map = 2.0 + (98.0 * (1.0 - depth_array))
            estimator.close()
            return depth_map.astype(np.float32)
        except Exception as exc:
            print(f"MediaPipe depth failed: {exc}.")
            return None
    
class FallbackDepthEstimator(DepthEstimator):
    """Fallback depth estimator using simple heuristics."""
    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        height, _ = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        y = np.arange(height, dtype=np.float32)
        dist_from_horizon = y - self.cy

        lower_far_mask = (dist_from_horizon > 0) & (dist_from_horizon <= self.height * 0.3)
        lower_near_mask = dist_from_horizon > self.height * 0.3
        upper_mask = dist_from_horizon <= 0

        base_depth = np.empty(height, dtype=np.float32)
        base_depth[lower_near_mask] = 3.0
        t_lower = dist_from_horizon[lower_far_mask] / (self.height * 0.3)
        base_depth[lower_far_mask] = 15.0 + t_lower * (3.0 - 15.0)
        t_upper = np.abs(dist_from_horizon[upper_mask]) / max(self.cy, 1.0)
        base_depth[upper_mask] = 15.0 + t_upper * (50.0 - 15.0)

        edges_vert = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=7))
        row_max = np.max(edges_vert, axis=1, keepdims=True)
        edges_vert_norm = np.divide(
            edges_vert,
            row_max + 1e-6,
            out=np.zeros_like(edges_vert),
            where=row_max > 1e-6,
        )
        edges_vert_smooth = cv2.GaussianBlur(edges_vert_norm, (51, 1), 0)
        depth_adjustment = 1.0 - np.clip(edges_vert_smooth, 0, 1) * 0.6

        depth_map = base_depth[:, None] * depth_adjustment
        depth_map = cv2.GaussianBlur(depth_map, (21, 5), 1.0)
        depth_map = np.clip(depth_map, 0.5, 100.0).astype(np.float32)
        return depth_map

class VehicleDetector:

    def __init__(self,
        use_vehicle_calibration: bool = True,
        vehicle_heights: Optional[Dict[str, float]] = None,
        vehicle_min_score: float = 0.35,
        vehicle_max_results: int = 12,
        vehicle_min_bbox_pixels: int = 30):
        self.use_vehicle_calibration = use_vehicle_calibration
        self.vehicle_heights = dict(VEHICLE_HEIGHT_PRESETS_M)
        if vehicle_heights:
            self.vehicle_heights.update(vehicle_heights)
        self.vehicle_min_score = vehicle_min_score
        self.vehicle_max_results = vehicle_max_results
        self.vehicle_min_bbox_pixels = vehicle_min_bbox_pixels

    def _get_object_detector_model(self) -> str:
        """Download the MediaPipe object detector model on first use."""
        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "efficientdet_lite0.tflite"
        if not model_path.exists():
            print("Downloading MediaPipe object detector model...")
            urllib.request.urlretrieve(OBJECT_DETECTOR_MODEL_URL, model_path)
        return str(model_path)

    """Placeholder for a vehicle detection model loader."""
    def detect_vehicles(self, image: np.ndarray) -> List[VehicleDetection]:
        """Detect car, bus, and truck boxes used as metric references."""
        if not MEDIAPIPE_AVAILABLE:
            return []
        try:
            model_path = self._get_object_detector_model()
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                max_results=self.vehicle_max_results,
                score_threshold=self.vehicle_min_score,
                category_allowlist=list(VEHICLE_LABELS),
            )
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            with vision.ObjectDetector.create_from_options(options) as detector:
                result = detector.detect(mp_image)

            detections: List[VehicleDetection] = []
            for detection in result.detections:
                if not detection.categories:
                    continue
                category = detection.categories[0]
                label = category.category_name
                if label not in VEHICLE_LABELS:
                    continue
                bbox = detection.bounding_box
                x1 = max(0, int(bbox.origin_x))
                y1 = max(0, int(bbox.origin_y))
                x2 = min(self.width - 1, int(bbox.origin_x + bbox.width))
                y2 = min(self.height - 1, int(bbox.origin_y + bbox.height))
                if x2 <= x1 or y2 <= y1:
                    continue
                if (y2 - y1) < self.vehicle_min_bbox_pixels or (x2 - x1) < self.vehicle_min_bbox_pixels:
                    continue
                detections.append(
                    VehicleDetection(
                        label=label,
                        score=float(category.score),
                        bbox=(x1, y1, x2, y2),
                        assumed_height_m=float(self.vehicle_heights[label]),
                    )
                )
            detections.sort(key=lambda item: item.score, reverse=True)
            return detections
        except Exception as exc:
            print(f"Vehicle detection failed: {exc}")
            return []

class StreetViewElevationPipeline:
    """
    Estimate a Street View elevation map and draw water/elevation lines.

    Pipeline stages:
    1. Load image and compute camera geometry.
    2. Estimate monocular depth.
    3. Detect vehicles and calibrate metric scale/ground anchors.
    4. Reconstruct a ground-relative elevation map.
    5. Extract and draw elevation contours.
    """

    def __init__(
        self,
        # image: cv2.Mat,
        # fov_degrees: float = 50.0,
        camera_height: float = 1.5,
        use_depth: bool = True,
        use_vehicle_calibration: bool = True,
        vehicle_heights: Optional[Dict[str, float]] = None,
        vehicle_min_score: float = 0.35,
        vehicle_max_results: int = 12,
        vehicle_min_bbox_pixels: int = 30,
        output_file_name: Optional[str] = "output.jpg",
    ):
        # self.fov_degrees = fov_degrees
        self.camera_height = camera_height
        self.use_vehicle_calibration = use_vehicle_calibration
        self.vehicle_heights = dict(VEHICLE_HEIGHT_PRESETS_M)
        if vehicle_heights:
            self.vehicle_heights.update(vehicle_heights)
        self.vehicle_min_score = vehicle_min_score
        self.vehicle_max_results = vehicle_max_results
        self.vehicle_min_bbox_pixels = vehicle_min_bbox_pixels

        self.estimator = None
        self.camera_geometry = None

        # if use_depth:
        #     self._estimate_depth()
        #     if self.depth_map is not None and self.use_vehicle_calibration:
        #         self._calibrate_depth_from_vehicles()
        #     self._reconstruct_elevation_map()

    def process_image(self, image: np.ndarray, fov_degrees: float = 50.0):
        """Run the full pipeline on the input image."""
        elevation_map = ElevationMap()
        elevation_map.height, _ = image.shape[:2]
        self._calculate_camera_intrinsics(image, fov_degrees)
        elevation_map.depth_map = self._estimate_depth(image)
        if elevation_map.depth_map is not None and self.use_vehicle_calibration:
            self._calibrate_depth_from_vehicles(elevation_map.depth_map)
            self._reconstruct_elevation_map()

    def _estimate_depth(self, image: np.ndarray) -> list[np.ndarray]:
        """Run monocular depth estimation on the input image."""
        if self.estimator is None:
            self.estimator = self._get_depth_estimator(image)
        estimator = self.estimator
        if self.estimator is None:
            print("No depth estimator available.")
            return
        return self.estimator.estimate_depth(image)

    def _calculate_camera_intrinsics(self, image: cv2.Mat, fov_degrees: float):
        """Calculate camera intrinsics from horizontal FOV."""
        self.camera_geometry = CameraGeometry(image, fov_degrees)


    # Depth estimation -------------------------------------------------

    def _get_depth_estimator(self, image) -> DepthEstimator:
        try:
            midas_estimator = MiDaSDepthEstimator()
            if midas_estimator.model is not None:
                return midas_estimator
        except Exception as exc:
            print(f"MiDaS depth estimator initialization failed: {exc}")
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_estimator = MediaPipeDepthEstimator()
                test_depth = mp_estimator.estimate_depth(image)
                if test_depth is not None:
                    return mp_estimator
            except Exception as exc:
                print(f"MediaPipe depth estimator initialization failed: {exc}")
        print("Using fallback depth estimator.")    
        return FallbackDepthEstimator()

    # Vehicle calibration ----------------------------------------------

    def _get_object_detector_model(self) -> str:
        """Download the MediaPipe object detector model on first use."""
        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "efficientdet_lite0.tflite"
        if not model_path.exists():
            print("Downloading MediaPipe object detector model...")
            urllib.request.urlretrieve(OBJECT_DETECTOR_MODEL_URL, model_path)
        return str(model_path)


    def _is_reliable_calibration_detection(self, detection: VehicleDetection) -> bool:
        """Filter detections before using them as metric anchors."""
        x1, y1, x2, y2 = detection.bbox
        border_margin_x = max(3, int(self.width * 0.03))
        border_margin_y = max(3, int(self.height * 0.02))
        if (
            x1 <= border_margin_x
            or x2 >= self.width - 1 - border_margin_x
            or y1 <= border_margin_y
            or y2 >= self.height - 1 - border_margin_y
        ):
            return False
        return detection.score >= max(self.vehicle_min_score, 0.35)

    def _sample_depth_band(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        y_center: int,
        band_half_height: int = 2,
    ) -> Optional[float]:
        """Sample a robust median depth across a narrow horizontal band."""
        if depth_map is None:
            return None
        x1, _, x2, _ = bbox
        x_pad = max(1, int(0.1 * (x2 - x1)))
        x_start = max(0, x1 + x_pad)
        x_end = min(self.width, x2 - x_pad)
        if x_end <= x_start:
            x_start, x_end = x1, x2

        y_start = max(0, y_center - band_half_height)
        y_end = min(self.height, y_center + band_half_height + 1)
        band = depth_map[y_start:y_end, x_start:x_end]
        valid = band[np.isfinite(band) & (band > 0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _interpolate_anchor_profile(
        self,
        anchors: List[GroundAnchor],
        value_getter,
    ) -> Optional[np.ndarray]:
        """Interpolate a per-column anchor-derived profile."""
        if not anchors:
            return None
        anchors = sorted(anchors, key=lambda item: item.x)
        x_positions = np.array([item.x for item in anchors], dtype=np.float32)
        values = np.array([value_getter(item) for item in anchors], dtype=np.float32)
        if len(anchors) == 1:
            return np.full(self.width, values[0], dtype=np.float32)
        columns = np.arange(self.width, dtype=np.float32)
        profile = np.interp(columns, x_positions, values).astype(np.float32)
        kernel = max(5, (self.width // 40) | 1)
        profile = cv2.GaussianBlur(profile.reshape(1, -1), (kernel, 1), 0).reshape(-1)
        return profile.astype(np.float32)

    def _build_calibration_candidate(self, detection: VehicleDetection) -> Optional[CalibrationCandidate]:
        """Convert a detection into a trusted calibration candidate if possible."""
        if not self._is_reliable_calibration_detection(detection):
            return None
        _, _, vertical_angle, _ = self._get_camera_geometry()
        x1, y1, x2, y2 = detection.bbox
        bbox_height_px = y2 - y1
        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        top_depth = self._sample_depth_band(detection.bbox, y1)
        bottom_depth = self._sample_depth_band(detection.bbox, y2)
        center_depth = self._sample_depth_band(detection.bbox, center_y)
        if (
            bbox_height_px <= 0
            or center_depth is None
            or bottom_depth is None
            or center_depth <= 1e-6
            or bottom_depth <= 1e-6
        ):
            return None

        expected_depth_m = (self.fy * detection.assumed_height_m) / bbox_height_px
        if expected_depth_m > 40.0:
            return None
        scale = expected_depth_m / center_depth
        if not np.isfinite(scale) or not (0.05 <= scale <= 20.0):
            return None

        scaled_bottom_depth = bottom_depth * scale
        bottom_angle = vertical_angle[min(self.height - 1, y2), center_x]
        estimated_camera_height = -scaled_bottom_depth * np.sin(bottom_angle)
        if not np.isfinite(estimated_camera_height) or not (0.2 <= estimated_camera_height <= 6.0):
            return None

        return CalibrationCandidate(
            detection=detection,
            bbox_height_px=bbox_height_px,
            expected_depth_m=float(expected_depth_m),
            scale=float(scale),
            estimated_camera_height_m=float(estimated_camera_height),
            center_x=center_x,
            ground_anchor_y=y2,
            top_depth_m=None if top_depth is None else float(top_depth * scale),
            center_depth_m=float(center_depth * scale),
            bottom_depth_m=float(bottom_depth * scale),
        )

    def _calibrate_depth_from_vehicles(self, elevation_map: ElevationMap):
        """Use detected vehicles as metric scale references."""
        elevation_map.vehicle_detections = self._detect_vehicle_bboxes()
        if not elevation_map.vehicle_detections:
            elevation_map.calibration_summary = CalibrationSummary(
                enabled=True,
                applied=False,
                reason="no_vehicle_detections",
            )
            return

        candidates = [
            candidate
            for candidate in (
                self._build_calibration_candidate(detection)
                for detection in elevation_map.vehicle_detections
            )
            if candidate is not None
        ]
        elevation_map.accepted_calibration_candidates = candidates
        if not candidates:
            elevation_map.calibration_summary = CalibrationSummary(
                enabled=True,
                applied=False,
                reason="no_valid_vehicle_calibrations",
                detections_found=len(elevation_map.vehicle_detections),
            )
            return

        combined_scale = float(np.median([candidate.scale for candidate in candidates]))
        combined_camera_height = float(
            np.median([candidate.estimated_camera_height_m for candidate in candidates])
        )
        if len(candidates) == 1 and combined_camera_height > 3.5:
            elevation_map.calibration_summary = CalibrationSummary(
                enabled=True,
                applied=False,
                reason="single_detection_implausible_camera_height",
                detections_found=len(self.vehicle_detections),
                candidate_camera_height_m=combined_camera_height,
            )
            return

        elevation_map.depth_map = elevation_map.depth_map * combined_scale
        self.camera_height = combined_camera_height

        final_ground_anchors: List[GroundAnchor] = []
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.detection.bbox
            center_x = candidate.center_x
            bottom_depth = self._sample_depth_band(candidate.detection.bbox, y2)
            if bottom_depth is None or bottom_depth <= 1e-6:
                continue
            bottom_angle = self.camera_geometry.vertical_angle[min(self.height - 1, y2), center_x]
            final_ground_anchors.append(
                GroundAnchor(
                    x=float(center_x),
                    y=float(y2),
                    ground_height=float(
                        combined_camera_height + bottom_depth * combined_scale * np.sin(bottom_angle)
                    ),
                    pixels_per_meter=float((y2 - y1) / max(candidate.detection.assumed_height_m, 1e-6)),
                )
            )

        elevation_map.local_ground_profile = self._interpolate_anchor_profile(
            final_ground_anchors, lambda anchor: anchor.ground_height
        )
        elevation_map.local_ground_y_profile = self._interpolate_anchor_profile(
            final_ground_anchors, lambda anchor: anchor.y
        )
        elevation_map.local_pixels_per_meter_profile = self._interpolate_anchor_profile(
            final_ground_anchors, lambda anchor: anchor.pixels_per_meter
        )
        elevation_map.calibration_summary = CalibrationSummary(
            enabled=True,
            applied=True,
            reason="vehicle_references_applied",
            detections_found=len(elevation_map.vehicle_detections),
            detections_used=len(candidates),
            depth_scale_factor=combined_scale,
            camera_height_m=combined_camera_height,
            local_ground_profile_created=elevation_map.local_ground_profile is not None,
            local_ground_y_profile_created=elevation_map.local_ground_y_profile is not None,
            ground_anchors=final_ground_anchors,
            per_detection=candidates,
        )

    # Elevation reconstruction -----------------------------------------

    def _reconstruct_elevation_map(self, elevation_map: ElevationMap):
        """Rebuild the metric elevation map from calibrated depth."""
        if elevation_map.depth_map is None:
            return
    
        depth = elevation_map.depth_map.astype(np.float32)
        absolute_height = self.camera_height + depth * self.camera_geometry.sin_vertical_angle

        if elevation_map.local_ground_profile is not None:
            elevation = absolute_height - elevation_map.local_ground_profile[None, :]
        else:
            elevation = absolute_height

        max_upward_angle = np.radians(30)
        min_downward_angle = np.radians(-60)
        depth_min = max(0.5, float(np.percentile(depth, 2)) * 0.5)
        depth_max = max(10.0, float(np.percentile(depth, 98)) * 1.5)
        valid_mask = (
            (self.camera_geometry.vertical_angle > min_downward_angle)
            & (self.camera_geometry.vertical_angle < max_upward_angle)
            & (depth > depth_min)
            & (depth < depth_max)
            & np.isfinite(elevation)
        )

        masked_elevation = np.where(valid_mask, elevation, np.nan).astype(np.float32)
        elevation_map.elevation_map = self._smooth_valid_elevation(masked_elevation, valid_mask)
        elevation_map.elevation_valid_mask = valid_mask
        elevation_map.point_cloud = {
            "depth": depth,
            "vertical_angle": self.camera_geometry.vertical_angle,
            "absolute_height": absolute_height,
            "local_ground_profile": elevation_map.local_ground_profile,
            "depth_min": depth_min,
            "depth_max": depth_max,
        }

    def _smooth_valid_elevation(self, elevation: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Apply a mask-aware smoothing pass to the elevation map."""
        filled = np.where(valid_mask, elevation, 0.0).astype(np.float32)
        weights = valid_mask.astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        blurred_sum = cv2.filter2D(filled, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        blurred_weights = cv2.filter2D(weights, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        smoothed = np.divide(
            blurred_sum,
            blurred_weights,
            out=np.full_like(filled, np.nan),
            where=blurred_weights > 0,
        )
        return smoothed

    # Contour extraction / rendering -----------------------------------

    def _expected_contour_y(self, x: int, elevation_map: ElevationMap, elevation_m: float) -> Tuple[Optional[float], Optional[float]]:
        """Estimate the likely contour row from vehicle-bottom anchors."""
        if elevation_map.local_ground_y_profile  is None or self.local_pixels_per_meter_profile is None:
            return None, None
        pixels_per_meter = float(self.local_pixels_per_meter_profile[x])
        expected_y = float(elevation_map.local_ground_y_profile[x] - elevation_m * pixels_per_meter)
        max_allowed_offset = max(20.0, 1.5 * pixels_per_meter)
        return expected_y, max_allowed_offset

    def _candidate_y_range(self, expected_y: Optional[float], max_allowed_offset: Optional[float]) -> Tuple[int, int]:
        """Reduce contour scanning to a local window when vehicle anchors exist."""
        if expected_y is None or max_allowed_offset is None:
            return 0, self.height - 2
        y_min = max(0, int(expected_y - max_allowed_offset) - 2)
        y_max = min(self.height - 2, int(expected_y + max_allowed_offset) + 2)
        return y_min, y_max

    def _collect_contour_points(
        self, elevation_map: ElevationMap,elevation_m: float, search_tolerance: float = 0.5
    ) -> Optional[np.ndarray]:
        """Find the contour crossing nearest the local vehicle-ground expectation."""
        if elevation_map.elevation_map is None or elevation_map.elevation_valid_mask is None or elevation_m <= 0.0:
            return None

        contour_points = []
        for x in range(self.width):
            column_elevation = elevation_map.elevation_map[:, x]
            column_valid = elevation_map.elevation_valid_mask[:, x]
            expected_y, max_allowed_offset = self._expected_contour_y(x, elevation_m)
            y_start, y_end = self._candidate_y_range(expected_y, max_allowed_offset)

            best_score = float("inf")
            best_y = None
            for y1_idx in range(y_start, y_end + 1):
                y2_idx = y1_idx + 1
                if y2_idx >= self.height:
                    break
                if not column_valid[y1_idx] or not column_valid[y2_idx]:
                    continue

                elev1 = column_elevation[y1_idx]
                elev2 = column_elevation[y2_idx]
                if not np.isfinite(elev1) or not np.isfinite(elev2):
                    continue
                if (elev1 - elevation_m) * (elev2 - elevation_m) > 0.0:
                    continue
                if abs(elev2 - elev1) <= 1e-9:
                    continue

                t = np.clip((elevation_m - elev1) / (elev2 - elev1), 0.0, 1.0)
                y_interp = y1_idx + t
                interp_elev = elev1 + t * (elev2 - elev1)
                if abs(interp_elev - elevation_m) > search_tolerance or interp_elev < 0.0:
                    continue

                score = y_interp if expected_y is None else abs(y_interp - expected_y)
                if score < best_score:
                    best_score = score
                    best_y = y_interp

            if best_y is None:
                continue
            if max_allowed_offset is not None and best_score > max_allowed_offset:
                continue
            contour_points.append([x, best_y])

        if not contour_points:
            return None
        return np.array(contour_points, dtype=np.float32)

    def _draw_contour_points(
        self,
        image: np.ndarray,
        contour_points: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> np.ndarray:
        """Draw a polyline from contour points."""
        for i in range(len(contour_points) - 1):
            pt1 = tuple(contour_points[i].astype(int))
            pt2 = tuple(contour_points[i + 1].astype(int))
            cv2.line(image, pt1, pt2, color, thickness)
        return image

    def draw_elevation_line(
        self,
        image: np.ndarray,
        elevation_map: ElevationMap,
        elevation_m: float,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3,
        search_tolerance: float = 0.5,
    ) -> np.ndarray:
        """Draw a single calibrated elevation contour."""
        result = image.copy()
        contour_points = self._collect_contour_points(elevation_map, elevation_m, search_tolerance)
        if contour_points is None:
            return result
        result = self._draw_contour_points(result, contour_points, color, thickness)
        mid_idx = len(contour_points) // 2
        label_x = max(10, min(int(contour_points[mid_idx, 0]) - 30, self.width - 100))
        label_y = max(20, int(contour_points[mid_idx, 1]) - 10)
        cv2.putText(
            result,
            f"{elevation_m:.2f}m",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        return result

    def draw_multiple_elevations(
        self,
        image: np.ndarray,
        elevations: List[float],
        colors: Optional[List[Tuple[int, int, int]]] = None,
        search_tolerance: float = 0.5,
    ) -> np.ndarray:
        """Draw multiple elevation contours."""
        result = image.copy()
        if self.elevation_map is None:
            return result
        if colors is None:
            colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0)]
        for i, elev in enumerate(elevations):
            contour_points = self._collect_contour_points(elev, search_tolerance)
            if contour_points is None:
                continue
            result = self._draw_contour_points(
                result,
                contour_points,
                colors[i % len(colors)],
                thickness=2,
            )
        return result

    # Outputs ----------------------------------------------------------

    def draw_vehicle_detections(self, elevation_map: ElevationMap, image: np.ndarray) -> np.ndarray:
        """Return a debug image showing detected vehicles."""
        result = image.copy()
        for detection in elevation_map.vehicle_detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{detection.label} {detection.score:.2f} ({detection.assumed_height_m:.1f}m)"
            cv2.putText(
                result,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
        return result

    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Visualize the current calibrated depth map."""
        if depth_map is None:
            raise RuntimeError("Depth map not available")
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)

    def visualize_elevation(self) -> np.ndarray:
        """Visualize the current elevation map."""
        if self.elevation_map is None:
            raise RuntimeError("Elevation map not available")
        display = np.nan_to_num(self.elevation_map, nan=0.0)
        elev_norm = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(elev_norm.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    def save(self, output_path: str, image: np.ndarray):
        """Save an image, creating the parent directory if necessary."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)

    def write_vehicle_calibration_summary(self, elevation_map: ElevationMap, output_path: str):
        """Write a human-readable summary of the calibration run."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"image={self.output_file_name}",
            f"vehicle_calibration_enabled={elevation_map.calibration_summary.enabled}",
            f"vehicle_calibration_applied={elevation_map.calibration_summary.applied}",
            f"reason={elevation_map.calibration_summary.reason}",
        ]
        if elevation_map.calibration_summary.depth_scale_factor is not None:
            lines.append(f"depth_scale_factor={elevation_map.calibration_summary.depth_scale_factor:.6f}")
        if elevation_map.calibration_summary.camera_height_m is not None:
            lines.append(f"camera_height_m={elevation_map.calibration_summary.camera_height_m:.6f}")
        lines.append(f"detections_found={elevation_map.calibration_summary.detections_found}")
        lines.append(f"detections_used={elevation_map.calibration_summary.detections_used}")

        for i, candidate in enumerate(elevation_map.calibration_summary.per_detection, start=1):
            lines.append(
                f"detection_{i}={candidate.detection.label} "
                f"score={candidate.detection.score:.3f} "
                f"bbox={candidate.detection.bbox} "
                f"expected_depth_m={candidate.expected_depth_m:.3f} "
                f"scale={candidate.scale:.3f} "
                f"camera_height_m={candidate.estimated_camera_height_m:.3f}"
            )

        output.write_text("\n".join(lines) + "\n", encoding="utf-8")


StreetviewElevationLineDrawer = StreetViewElevationPipeline


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-show", "--show", action="store_true")
    parser.add_argument("-height", "--height", type=float, default=0.5)
    parser.add_argument(
        "-image",
        "--image",
        type=str,
        default="testImage.jpeg",
        help="Path to input Street View image",
    )
    parser.add_argument("--fov-degrees", type=float, default=50.0)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument(
        "--disable-vehicle-calibration",
        action="store_true",
        help="Skip vehicle-based metric calibration and use the raw depth scaling.",
    )
    parser.add_argument("--vehicle-min-score", type=float, default=0.35)
    parser.add_argument("--vehicle-max-results", type=int, default=12)
    parser.add_argument("--vehicle-min-bbox-pixels", type=int, default=30)
    parser.add_argument("--car-height-m", type=float, default=VEHICLE_HEIGHT_PRESETS_M["car"])
    parser.add_argument("--bus-height-m", type=float, default=VEHICLE_HEIGHT_PRESETS_M["bus"])
    parser.add_argument("--truck-height-m", type=float, default=VEHICLE_HEIGHT_PRESETS_M["truck"])
    args = parser.parse_args()

    print(f"Using water level: {args.height:.2f}m")
    print(f"Input image: {args.image}")
    return args

def draw_elevation_line_from_buffer(imageBuffer: np.ndarray, water_level_meters: float) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(imageBuffer, np.uint8), cv2.IMREAD_COLOR)
    drawer = StreetviewElevationLineDrawer(
        image,
        fov_degrees=50.0,
        camera_height=1.5,
        use_depth=True,
        use_vehicle_calibration=True,
        vehicle_heights={
            "car": VEHICLE_HEIGHT_PRESETS_M["car"],
            "bus": VEHICLE_HEIGHT_PRESETS_M["bus"],
            "truck": VEHICLE_HEIGHT_PRESETS_M["truck"],
        },
        vehicle_min_score=0.35,
        vehicle_max_results=12,
        vehicle_min_bbox_pixels=30,
    )
    result = drawer.draw_elevation_line(water_level_meters)
    return cv2.imencode(".jpg", result)[1].tobytes()

if __name__ == "__main__":
    args = parse_arguments()
    water_level_meters = args.height
    vehicle_heights = {
        "car": args.car_height_m,
        "bus": args.bus_height_m,
        "truck": args.truck_height_m,
    }
    image = cv2.imread(args.image)
    drawer = StreetviewElevationLineDrawer(
        image,
        fov_degrees=args.fov_degrees,
        camera_height=args.camera_height,
        use_depth=True,
        use_vehicle_calibration=not args.disable_vehicle_calibration,
        vehicle_heights=vehicle_heights,
        vehicle_min_score=args.vehicle_min_score,
        vehicle_max_results=args.vehicle_max_results,
        vehicle_min_bbox_pixels=args.vehicle_min_bbox_pixels,
        output_file_name=args.image,
    )
    outname = os.path.splitext(os.path.basename(args.image))[0]

    if drawer.depth_map is not None:
        print("\nDepth map analysis:")
        print(f"  Range: {drawer.depth_map.min():.1f}m - {drawer.depth_map.max():.1f}m")
        mid_row = drawer.height // 2
        left_depth = drawer.depth_map[mid_row, : drawer.width // 4].mean()
        center_depth = drawer.depth_map[
            mid_row, drawer.width // 3 : 2 * drawer.width // 3
        ].mean()
        right_depth = drawer.depth_map[mid_row, -drawer.width // 4 :].mean()
        print(f"  At mid-row (y={mid_row}):")
        print(f"    Left avg:   {left_depth:.1f}m")
        print(f"    Center avg: {center_depth:.1f}m")
        print(f"    Right avg:  {right_depth:.1f}m")

    print("\nCalibration status:")
    summary = drawer.calibration_summary
    for key in [
        "enabled",
        "applied",
        "reason",
        "detections_found",
        "detections_used",
        "depth_scale_factor",
        "camera_height_m",
        "candidate_camera_height_m",
        "local_ground_profile_created",
        "local_ground_y_profile_created",
    ]:
        value = getattr(summary, key)
        if value is not None:
            print(f"  {key}: {value}")

    if drawer.elevation_map is not None:
        print(f"\nElevation map analysis at target {water_level_meters}m:")
        probe_x = min(60, drawer.width - 1)
        print(f"  Elevation across rows (left side, x={probe_x}):")
        col_probe = drawer.elevation_map[:, probe_x]
        for y in [50, 100, 150, 200, 250, 300, 350]:
            if y < drawer.height:
                print(f"    y={y}: {col_probe[y]:.2f}m")

        print(f"\n  Finding where elevation = {water_level_meters}m:")
        for x_frac in [0.1, 0.5, 0.9]:
            x = int(x_frac * drawer.width)
            col_elev = drawer.elevation_map[:, x]
            valid_matches = np.isfinite(col_elev)
            matches = np.where(valid_matches & (np.abs(col_elev - water_level_meters) < 0.3))[0]
            if len(matches) > 0:
                print(
                    f"    x={x:4d}: elevation ≈ {water_level_meters}m "
                    f"at rows {matches[:5]}"
                )
            else:
                if np.any(valid_matches):
                    idx = np.nanargmin(np.abs(col_elev - water_level_meters))
                    print(f"    x={x:4d}: closest elevation is {col_elev[idx]:.2f}m at y={idx}")
                else:
                    print(f"    x={x:4d}: no valid elevation samples")

    result = drawer.draw_elevation_line(water_level_meters)
    drawer.save(f"output/{outname}.jpeg", result)
    print(f"Saved output/{outname}.jpeg")

    if drawer.depth_map is not None and drawer.elevation_map is not None:
        print("\nEdge depth analysis:")
        edges_to_check = [(10, "left-edge"), (drawer.width - 10, "right-edge")]
        for x, label in edges_to_check:
            x = int(np.clip(x, 0, drawer.width - 1))
            col_depth = drawer.depth_map[:, x]
            col_elev = drawer.elevation_map[:, x]
            print(f"  {label} (x={x}):")
            print(
                f"    Depth: min={col_depth.min():.1f}m, "
                f"max={col_depth.max():.1f}m, std={np.std(col_depth):.1f}m"
            )
            print(
                f"    Elevation: min={np.nanmin(col_elev):.2f}m, "
                f"max={np.nanmax(col_elev):.2f}m"
            )

    if drawer.depth_map is not None:
        drawer.save(f"output/{outname}_depth.jpg", drawer.visualize_depth())
        drawer.save(f"output/{outname}_elevation.jpg", drawer.visualize_elevation())
        print(f"Saved output/{outname}_depth.jpg and output/{outname}_elevation.jpg")

    if drawer.vehicle_detections:
        drawer.save(f"output/{outname}_detections.jpg", drawer.draw_vehicle_detections())
        print(f"Saved output/{outname}_detections.jpg")
    drawer.write_vehicle_calibration_summary(f"output/{outname}_vehicle_calibration.txt")
    print(f"Saved output/{outname}_vehicle_calibration.txt")

    if args.show:
        cv2.imshow("Elevation Line", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
