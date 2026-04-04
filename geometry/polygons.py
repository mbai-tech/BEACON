import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate

def make_robot_polygon(shape="rectangle"):
    """Returns a Shapely Polygon centered at origin for the robot."""
    if shape == "rectangle":
        return Polygon([(-0.3, -0.2), (0.3, -0.2), (0.3, 0.2), (-0.3, 0.2)])
    elif shape == "L":
        return Polygon([(0,0),(0.4,0),(0.4,0.2),(0.2,0.2),(0.2,0.4),(0,0.4)])

def transform_polygon(poly, x, y, theta_deg):
    """Move and rotate a polygon to pose (x, y, theta)."""
    rotated = rotate(poly, theta_deg, origin=(0, 0), use_radians=False)
    return translate(rotated, x, y)

def contact_area(robot_poly, obstacle_poly):
    """Area of overlap between robot and obstacle (0 if no contact)."""
    return robot_poly.intersection(obstacle_poly).area

def min_distance(robot_poly, obstacle_poly):
    """Minimum distance between robot and obstacle boundaries."""
    return robot_poly.distance(obstacle_poly)