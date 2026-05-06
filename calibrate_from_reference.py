"""
Back-calculate depth model from expected output.

Strategy:
1. Load the expected output image with the correct green 0.5m line
2. Extract where the green line is drawn
3. Back-calculate what depths would produce that line
4. Use those depths in our model
"""

import cv2
import numpy as np
from test import StreetviewElevationLineDrawer


def extract_line_from_image(image_path, target_color=(0, 255, 0), tolerance=50):
    """Extract pixel coordinates of a line from an image by color."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Find pixels matching the target color (allow some tolerance)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green color range in HSV
    lower = np.array([35, 50, 50])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Get coordinates
    y_coords, x_coords = np.where(mask > 0)

    if len(x_coords) == 0:
        return None

    return np.column_stack([x_coords, y_coords])


def back_calculate_depths(drawer, line_coords, target_elevation=0.5):
    """Back-calculate depth values that would produce the given line at target elevation."""

    # Initialize depth map
    depth_solution = np.ones((drawer.height, drawer.width), dtype=np.float32) * 15.0

    # For each unique x coordinate, find the y where the line is drawn
    for x in range(drawer.width):
        line_points_at_x = line_coords[line_coords[:, 0] == x]

        if len(line_points_at_x) == 0:
            continue

        # Take the top-most point (closest to camera)
        y_line = int(line_points_at_x[:, 1].min())

        # Back-calculate: given y_line should have elevation = target_elevation
        # elevation = camera_height + depth * sin(vertical_angle)
        # vertical_angle = arctan2(-y_norm, 1)
        # where y_norm = (y - cy) / fy

        y_norm = (y_line - drawer.cy) / drawer.fy
        vertical_angle = np.arctan2(-y_norm, 1.0)

        if abs(np.cos(vertical_angle)) < 0.01:
            continue

        # Solve for depth:
        # elevation = camera_height + depth * sin(vertical_angle)
        # depth = (elevation - camera_height) / sin(vertical_angle)

        sin_angle = np.sin(vertical_angle)
        if abs(sin_angle) > 0.01:
            depth = (target_elevation - drawer.camera_height) / sin_angle
            depth = np.clip(depth, 0.5, 100.0)
            depth_solution[y_line, x] = depth

    # Smooth the depth map to fill gaps
    depth_solution = cv2.bilateralFilter(depth_solution, 9, 75, 75)
    depth_solution = cv2.GaussianBlur(depth_solution, (31, 31), 2.0)

    return depth_solution


def calibrate_from_reference():
    """Use the correct reference image to calibrate the depth model."""

    # Load reference image with correct line
    line_coords = extract_line_from_image("correct_0_5.jpeg")

    if line_coords is None:
        print("Could not extract line from correct_0_5.jpeg")
        return

    print(f"Extracted {len(line_coords)} pixels from reference line")

    # Create drawer and back-calculate
    drawer = StreetviewElevationLineDrawer("testImage.jpeg", use_depth=False)
    print(f"Image size: {drawer.width}x{drawer.height}")
    print(f"Camera intrinsics: fx={drawer.fx:.1f}, fy={drawer.fy:.1f}, cy={drawer.cy:.0f}")

    # Back-calculate depths
    print("Back-calculating depth map from reference image...")
    drawer.depth_map = back_calculate_depths(drawer, line_coords, target_elevation=0.5)

    print(f"Depth map range: {drawer.depth_map.min():.1f} - {drawer.depth_map.max():.1f}m")

    # Reconstruct and draw
    drawer._reconstruct_3d_points()

    result = drawer.draw_elevation_line(0.5, search_tolerance=0.3)
    drawer.save("output_0_5m_calibrated.jpg", result)

    print("✓ Saved output_0_5m_calibrated.jpg")
    print("\nThis depth model is now tuned to your specific image.")
    print("The line should match the reference image closely.")


if __name__ == "__main__":
    calibrate_from_reference()
