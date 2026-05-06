import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def look_at(camera_pos, target, up=np.array([0,0,1])):
    forward = (target - camera_pos)
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)

    R = np.vstack([right, up, forward])
    return R


def setup_camera(center_elev, image_height, image_width, fov_deg=90, radius=50, look_direction=270):
    camera_pos = np.array([
        0,
        -radius,
        center_elev
    ])

    # Convert look direction to radians
    look_rad = math.radians(look_direction)
    target = camera_pos + np.array([
        math.cos(look_rad),
        math.sin(look_rad),
        -0.0
    ]) 

    R = look_at(camera_pos, target)

    # Camera intrinsics
    fov = math.radians(fov_deg)

    fx = (image_width / 2) / math.tan(fov / 2) 
    fy = fx  # assume square pixels
    cx = image_width / 2
    cy = image_height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return camera_pos, R, K

def create_elevation_circle(camera_pos, radius, elevation):
    num_points = 500

    # elevation = elevation - 20  # adjust elevation to be relative to camera height

    angles = np.linspace(0, 2*np.pi, num_points)

    print(f"Creating circle with radius {radius} m at elevation {elevation} m")

    circle_points = []

    for theta in angles:
        x = camera_pos[0] + radius * np.cos(theta)
        y = camera_pos[1] + radius * np.sin(theta)
        z = elevation  # ground plane offset 

        circle_points.append([x, y, z])

    return np.array(circle_points)

def project_circle_to_image(circle_points_world, camera_pos, R, K, image_width, image_height):
    points_cam = (R @ (circle_points_world.T - camera_pos.reshape(3,1)))
    valid = points_cam[2] > 0

    points_cam = points_cam[:, valid]
    points_world_valid = circle_points_world[valid]

    points_img = K @ points_cam
    points_img /= points_img[2]

    u = points_img[0].astype(int)
    v = image_height - points_img[1].astype(int)
    z = points_cam[2]

    # Clip to image
    valid = (
        (u >= 0) & (u < image_width) &
        (v >= 0) & (v < image_height)
    )

    u = u[valid]
    v = v[valid]

    return u, v, z, points_world_valid

def project_waterlevel_contour_to_image_from_buffer(water_level_meters, street_view_img, line_color =(255, 0, 0), camera_height_agl_meters = 1.5, look_direction_degrees = 270):
    img = cv2.imdecode(np.frombuffer(street_view_img, np.uint8), cv2.IMREAD_COLOR)
    return project_waterlevel_contour_to_image(water_level_meters, img, line_color, camera_height_agl_meters, look_direction_degrees)


def project_waterlevel_contour_to_image(water_level_meters, image, line_color =(255, 0, 0), camera_height_agl_meters = 1.5, look_direction_degrees = 270):
    h, w = image.shape[:2]

    # Debug print the input parameters
    print(f"Projecting water level contour with parameters:")
    print(f"  Water level (m): {water_level_meters}")
    print(f"  Camera height above ground (m): {camera_height_agl_meters}")
    print(f"  Look direction (degrees): {look_direction_degrees}")
    print(f"  Image dimensions: {w}x{h}")

    # Setup the camera
    camera_pos, R, K = setup_camera(camera_height_agl_meters, h, w, look_direction=look_direction_degrees)

    # Create a circle around the center point then project the circle onto the image to verify camera setup
    elevation = (camera_height_agl_meters * -1) + water_level_meters
    circle_points_world = create_elevation_circle(camera_pos=camera_pos,radius=50, elevation=elevation) 

    # Project to the image
    u, v, _, _ = project_circle_to_image(circle_points_world, camera_pos, R, K, w, h)

    # Draw it
    pts = np.vstack([u, v]).T.astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    cv2.polylines(
        image,
        [pts],
        isClosed=False,
        color=line_color,
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return cv2.imencode('.jpg', image)[1].tobytes()

if __name__ == "__main__": 
    # Test Inputs
    water_level_meters = 1.5
    camera_height_above_ground = 1.5
    look_direction_degrees = 270  # looking west
    image = cv2.imread("testImage.jpeg")
    color = (255, 0, 0)  # Blue color for the contour
    h, w = image.shape[:2]


    project_waterlevel_contour_to_image(water_level_meters, image, line_color=color, camera_height_agl_meters=camera_height_above_ground, look_direction_degrees=look_direction_degrees)

    # Display the image to confirm placement
    cv2.imshow(f"Projected {water_level_meters} m contour", image)
    cv2.waitKey(0)