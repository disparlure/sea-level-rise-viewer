"""
Interactive depth calibration tool.

Usage: User marks known distances in the scene, and the script adjusts the depth model.
This lets us get reasonable elevation lines without needing a full depth estimator.
"""

import cv2
import numpy as np
from test import StreetviewElevationLineDrawer


def calibrate_by_features(image_path):
    """
    Calibrate depth based on image features and assumptions about street layout.

    For a typical urban street scene:
    - Ground level: 0m (or below camera at 1.5m)
    - Nearby walls/structures: 1-5m away
    - Mid-distance buildings: 10-30m
    - Far horizon: 50m+
    """

    drawer = StreetviewElevationLineDrawer(image_path, use_depth=False)

    print("Manual depth calibration mode")
    print("=" * 50)
    print("\nLooking at your street image, we need to calibrate distances.")
    print("\nFor a typical street scene, estimate:")
    print("  - Nearest visible wall/ground: ___ meters")
    print("  - Mid-distance buildings: ___ meters")
    print("  - Horizon/far buildings: ___ meters")
    print()

    # Use reasonable defaults based on typical street view
    near_depth = 3.0  # Close structures
    mid_depth = 15.0  # Mid-distance
    far_depth = 50.0  # Horizon

    print(f"Using defaults: near={near_depth}m, mid={mid_depth}m, far={far_depth}m")
    print("(Adjust these in the source code if the output doesn't look right)")

    # Build depth map based on vertical position
    h_3rd = drawer.height // 3

    drawer.depth_map = np.zeros((drawer.height, drawer.width), dtype=np.float32)

    # Bottom third: near structures
    drawer.depth_map[2*h_3rd:, :] = near_depth

    # Middle third: mid-distance
    drawer.depth_map[h_3rd:2*h_3rd, :] = mid_depth

    # Top third: far/sky
    drawer.depth_map[:h_3rd, :] = far_depth

    # Smooth transitions
    drawer.depth_map = cv2.GaussianBlur(drawer.depth_map, (21, 21), 2.0)

    # Reconstruct 3D and draw
    drawer._reconstruct_3d_points()

    result = drawer.draw_elevation_line(0.5)
    drawer.save("output_0_5m_calibrated.jpg", result)

    print("\nSaved: output_0_5m_calibrated.jpg")
    print("\nIf the line still doesn't match expectations,")
    print("adjust near_depth, mid_depth, far_depth values above and re-run.")


if __name__ == "__main__":
    calibrate_by_features("testImage.jpeg")
