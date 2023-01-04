import numpy as np
import math
import cv2

def get_distance(p1: list, p2: list):
    """
    Compute the euclidean distance between two points
    args:
        p1: point 1, list of two elements [x, y]
        p2: point 2, list of two elements [x, y]
    return:
        distance: euclidean distance between p1 and p2
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def get_centroid(points: list):
    """
    Compute the centroid of a list of points
    args:
        points: list of points, each point is a list of two elements
    return:
        centroid: centroid of the points, list of two elements [x, y]
    """
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid


def get_angle(a: list, b: list, c: list):
    """
    Calculate the angle between three points
    args:
        a: point a
        b: point b
        c: point c
    return:
        angle: angle between a-b-c
    """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)


def get_length_third_side(a, b, alpha):
    """
    Given the length of two sides of a trinagle and the angle between them, compute the length of the third side
    args:
        a: length of the first side
        b: length of the second side
        alpha: angle between the two sides
    return:
        c: length of the third side
    """

    # Reference:
    # - https://math.stackexchange.com/questions/134012/two-sides-and-angle-between-them-triangle-question
    # - https://www.calculatorsoup.com/calculators/geometry-plane/triangle-theorems.php

    # Side, Angle, Side
    # Given the size of 2 sides (a and b) and the size of the angle ALPHA that is in between those 2 sides you
    # can calculate the sizes of the remaining 1 side and 2 angles.
    # - Use The Law of Cosines to solve for the remaining side, c
    # - Determine which side, a or b, is smallest and use the Law of Sines to solve for the size of the opposite
    #   angle, BETA or GAMMA respectively.
    # - Use the Sum of Angles Rule to find the last angle

    # Convert alpha to radians
    alpha = math.radians(alpha)
    # Law of Cosines
    c = math.sqrt(a ** 2 + b ** 2 - 2 * a * b * math.cos(alpha))
    if a < b:
        beta = math.asin(a * math.sin(alpha) / c)
        gamma = math.pi - alpha - beta
    else:
        gamma = math.asin(b * math.sin(alpha) / c)
        beta = math.pi - alpha - gamma

    # Convert all angles to degrees
    alpha = math.degrees(alpha)
    beta = math.degrees(beta)
    gamma = math.degrees(gamma)
    
    return c


def get_parallel_line(p1, p2, p3):
    """
    Find the parallel line of the segment p1-p2, that passes through p3
    args:
        p1: the first point of the segment
        p2: the second point of the segment
        p3: the point that the parallel line must pass through
    returns:
        p4: the second point of the parallel line
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # To find the parallel segment, we need to first determine the slope of the original line.
    m = (y2 - y1) / (x2 - x1)

    # Once we have the slope, we can use the point-slope formula to find the equation of the
    # line that passes through point (x3, y3) with the same slope:
    # y - y3 = m * (x - x3)
    # y = m * x - m * x3 + y3

    # Choose x4
    x4 = x3 + 100
    y4 = int(m * x4 - m * x3 + y3)

    return (x3, y3), (x4, y4)


def get_perpendicular_line(wheels_center, com):
    """
    Compute the perpendicular segment between the wheels center and the center of mass
    args:
        wheels_center: the center of the wheels
        com: the center of mass
        image: the image where the line will be printed
    returns:
        p1: the first point of the perpendicular segment
        p2: the second point of the perpendicular segment
    """
    wheel_a, wheel_b = wheels_center
    x1, y1 = wheel_a
    x2, y2 = wheel_b
    x3, y3 = com
    k = ((y2 - y1) * (x3 - x1) - (x2 - x1) * (y3 - y1)) / ((y2 - y1) ** 2 + (x2 - x1) ** 2)
    x4 = int(x3 - k * (y2 - y1))
    y4 = int(y3 + k * (x2 - x1))
    return (x3, y3), (x4, y4)


def get_center_of_mass(image, segmentation_mask):
    """Get the center of mass of the cyclist
        args:
            image: the image where the center of mass will be computed
        returns:
            center_of_mass: the center of mass point of the cyclist
    """
    # calculate moments of binary image
    m = cv2.moments(segmentation_mask)
    # calculate x,y coordinate of center
    c_x = int(m["m10"] / m["m00"])
    c_y = int(m["m01"] / m["m00"])
    center_of_mass = (c_x, c_y)
    return center_of_mass
