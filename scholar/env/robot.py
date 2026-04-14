"""
robot.py — Robot model constants and helpers.

Uses the same ROBOT_RADIUS and disk-body model as NewProject.
"""

from NewProject.constants import ROBOT_RADIUS
from NewProject.planner import robot_body, normalize, clip_point_to_workspace

__all__ = ["ROBOT_RADIUS", "robot_body", "normalize", "clip_point_to_workspace"]
