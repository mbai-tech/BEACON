from geometry.polygons import make_robot_polygon, transform_polygon, contact_area
from shapely.geometry import Polygon

robot = make_robot_polygon("rectangle")
robot_moved = transform_polygon(robot, 1.0, 0.5, 45)
print("Area:", robot_moved.area)

obstacle = Polygon([(0.8, 0.3), (1.2, 0.3), (1.2, 0.7), (0.8, 0.7)])
print("Contact area:", contact_area(robot_moved, obstacle))