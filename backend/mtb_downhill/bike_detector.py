import mediapipe as mp
import os
import sys
import cv2
from backend.mtb_downhill.yolo_object_detection import *

sys.path.append("../../")
from config import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def detect_landmarks(image):
    """Detect the landmarks of the body
        args:   
            image: the image where the landmarks will be detected
        returns:
            landmarks: the landmarks of the body
    """
    # Determine width and height of the image
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:

        image_copy = image.copy()
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return False

        landmarks = results.pose_landmarks.landmark
        image_copy.flags.writeable = True
        mp_drawing.draw_landmarks(
            image_copy,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        

        if DEBUG:
            cv2.imshow('MediaPipe Pose', results.segmentation_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image_copy, landmarks, results.segmentation_mask

def wheels_under_hips(landmarks, circles):
    """
    Filter the circles that are under the hips.
    args:
        landmarks: the landmarks of the body
        circles: the circles detected in the image
    returns:
        circles: the circles that are under the hips
    """
    hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    hip = hip_right if hip_left.visibility < hip_right.visibility else hip_left

    new_circles = []
    for circle in circles:
        x, y, r = circle
        if y > hip.y:
            new_circles.append(circle)
    return new_circles


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


def check_if_point_is_between_two_points(point, point1, point2):
    """Check if a point is between two other points
        args:   
            point: the point to check
            point1: the first point
            point2: the second point
        returns:
            True if the point is between the two other points
            False if the point is not between the two other points
    """
    if point > point1 and point < point2:
        return True
    else:
        return False


def detect_wheels(image):
    """Detect the wheels of the bike
        args:   
            image: the image where the wheels will be detected
        returns:
            circles: the circles detected in the image
    """
    output = image.copy()
    image_height, image_width, _ = image.shape
    # 1. Convert the BGR image to RGB before processing.
    image_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # 2. To grayscale
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # 3. Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)
    if DEBUG:
        cv2.imshow('Blurred Image', img_blur)
        cv2.waitKey(0)


    # 4. Edge detection with Canny
    # Parameters:
    #   - image: 8-bit input image.
    #   - threshold1: first threshold for the hysteresis procedure.
    #   - threshold2: second threshold for the hysteresis procedure.
    edges = cv2.Canny(img_blur, 80, 180)
    if DEBUG:
        cv2.imshow('Edge Detection', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # detect circles in the image
    # Parameters: 
    #   - image: 8-bit, single-channel, grayscale input image.
    #   - method: Defines the method to detect circles in images.
    #   - dp: Inverse ratio of the accumulator resolution to the image resolution. 
    #         For example, if dp=1 , the accumulator has the same resolution as the input image. 
    #         If dp=2 , the accumulator has half as big width and height.
    #   - minDist: Minimum distance between the centers of the detected circles. 
    #              If the parameter is too small, multiple neighbor circles may be falsely 
    #              detected in addition to a true one. If it is too large, some circles may 
    #              be missed.
    #   - param1: First method-specific parameter. In case of cv2.HOUGH_GRADIENT , it is 
    #             the higher threshold of the two passed to the Canny edge detector (the lower 
    #             one is twice smaller).
    #   - param2: Second method-specific parameter. In case of cv2.HOUGH_GRADIENT , it is the 
    #             accumulator threshold for the circle centers at the detection stage. 
    #             The smaller it is, the more false circles may be detected. Circles, corresponding 
    #             to the larger accumulator values, will be returned first.
    #   - minRadius: Minimum circle radius.
    #   - maxRadius: Maximum circle radius.
    circles = cv2.HoughCircles(edges,
                                cv2.HOUGH_GRADIENT, 
                                dp=2, 
                                minDist=100, 
                                param1=90, 
                                param2=150, 
                                minRadius=0, 
                                maxRadius=200)
    # 6/7 images with: 
    #   - Canny: threshold1 = 80, threshold2 = 200
    #   - HoughCircle: dp=2, minDist=100, param1=90, param2=150, minRadius=0, maxRadius=200
    return circles


def check_crank_position(image, landmarks, circle_a, circle_b):
    """
    Check if the crank is in the right position, i.e. if the foot are in the same 
    horizontal line of the two wheels centers.
    args:
        landmarks: the landmarks of the cyclist
        circle_center_a: the center of the first wheel
        circle_center_b: the center of the second wheel
    returns:
        True if the crank is in the right position
        False if the crank is not in the right position
    """
    image = image.copy()
    #  Get image size
    height, width, channels = image.shape

    # Get the foot landmark
    foot_left = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    foot_right = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    # Get the heel landmark
    heel_left = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    heel_right = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

    # We can use only one foot landmark, the one with the highest visibility
    # because if one of the two foot is in the right position, the other one will be too
    
    # Compare the visibility of the landmarks
    if foot_left.visibility < foot_right.visibility:
        foot = foot_left
        heel = heel_left
    else:
        foot = foot_right
        heel = heel_right

    # Get middle point between foot and heel
    foot_x = (foot.x + heel.x) / 2
    foot_y = (foot.y + heel.y) / 2
    foot = (int(foot_x*width), int(foot_y*height))


    # Unpack the circle centers
    circle_a_x, circle_a_y, circle_a_r = circle_a
    circle_b_x, circle_b_y, circle_b_r = circle_b

    # Draw a green rectangle around the line between the two wheels to simulate a threshold
    # and check if the foot are inside the rectangle.
    # To compute the threshold, we take the 30% of the mean of the two wheels radius
    threshold = int((circle_a_r + circle_b_r) / 2 * 0.3)

    # Get two point 5 px above, and below the first wheel center
    wheel_a_above = (int(circle_a_x), int(circle_a_y - threshold))
    wheel_a_below = (int(circle_a_x), int(circle_a_y + threshold))
    # Do the same for the second wheel
    wheel_b_above = (int(circle_b_x), int(circle_b_y - threshold))
    wheel_b_below = (int(circle_b_x), int(circle_b_y + threshold))

    # Draw the line between each point
    cv2.line(image, wheel_a_above, wheel_a_below, (0, 255, 0), 2)
    cv2.line(image, wheel_a_below, wheel_b_below, (0, 255, 0), 2)
    cv2.line(image, wheel_b_below, wheel_b_above, (0, 255, 0), 2)
    cv2.line(image, wheel_b_above, wheel_a_above, (0, 255, 0), 2)

    # Select the green rectangle
    lower_upper = np.array([0, 255, 0])
    mask = cv2.inRange(image, lower_upper, lower_upper)
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Extract the contour of the rectangle to use the pointPolygonTest function
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 120, 255, 1)
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cv2.circle(image, foot, 5, (0, 0, 255), -1)
    
    if DEBUG:
        cv2.imshow("result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # PointPolygonTest function returns: 
    #   -1 if the point is outside the contour
    #   1 if the point is on the contour
    result = cv2.pointPolygonTest(cnts[0], foot, False)

    return True if result > 0 else False


def check_heel_position(landmarks):
    """
    Check if the heel is in the right position, i.e. if the heel is under the foot_index
    args:
        landmarks: the landmarks of the cyclist
    returns:
        True if the heel is in the right position
        False if the heel is not in the right position
    """
    # Get the foot landmark
    foot_left = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    foot_right = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    # Get the heel landmark
    heel_left = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    heel_right = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

    # Check visibility of the landmarks, if upper than 0.5, the landmark is visible
    # and we can use it
    left_position, right_position = None, None
    if foot_left.visibility > 0.3 and heel_left.visibility > 0.3:
        left_position = True if heel_left.y > foot_left.y else False
    if foot_right.visibility > 0.3 and heel_right.visibility > 0.3:
        right_position = True if heel_right.y > foot_right.y else False

    if DEBUG:
        print("RIGHT Visibility:", foot_right.visibility, heel_right.visibility)
        print("LEFT Visibility:", foot_left.visibility, heel_left.visibility)
        print("RIGHT Position:", right_position)
        print("LEFT Position:", left_position)
    return left_position, right_position


def get_bike_direction(landmarks: list):
    """
    Compute the direction of the bike
    args:
        landmarks_list: list of landmarks, one list for each frame, and each landmark is a list of 33 elements
    return:
        direction: direction of the bike, 'left' or 'right'
    """
    # Get WRIST and HIP landmarks
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Check visibility of the landmarks
    if left_wrist.HasField('visibility') and right_wrist.HasField('visibility'):
        # Assume that the shoulder of the same side as the hip is visible in a better way
        if left_wrist.visibility < right_wrist.visibility:
            wrist, hip = right_wrist, right_hip
        else:
            wrist, hip = left_wrist, left_hip
    # Check which is forward and return
    return BikeFittingEnum.LEFT.value if wrist.x < hip.x else BikeFittingEnum.RIGHT.value


def check_head_hand_distance(landmarks, circle_a, circle_b):
    """
    Check if the distance between the head and the hand is correct
    args:
        landmarks: the landmarks of the cyclist
        circle_a: the first wheel   
        circle_b: the second wheel
    returns:
        True if the head is correct
        False if the distance is not correct
    """
    # Get the hand landmark
    hand_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    hand_right = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    # get the hand with the highest visibility
    hand = hand_left if hand_left.visibility > hand_right.visibility else hand_right
    # Get a point of the face. In the list returned from Medipipe, the first
    # 10 point are on the face (ex. nose, left eye, right eye, etc.)
    # Loop over the first 10 landmarks to find the point with the highest visibility
    # and use it as the head landmark
    head = None
    for i in range(10):
        if head is None or head.visibility < landmarks[i].visibility:
            head = landmarks[i]
    
    mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
    mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
    mouth = mouth_left if mouth_left.visibility > mouth_right.visibility else mouth_right

    # Unpack the circle centers
    circle_a_x, circle_a_y, circle_a_r = circle_a
    circle_b_x, circle_b_y, circle_b_r = circle_b

    # Draw a green rectangle around the line between the two wheels to simulate a threshold
    # and check if the foot are inside the rectangle.
    # To compute the threshold, we take the 30% of the mean of the two wheels radius
    threshold = (circle_a_r + circle_b_r) / 2 * 0.05
    
    # If the head is around on the same vertical level of
    # the hand, the position is incorrect
    position = None

    side = get_bike_direction(landmarks)
    if side == BikeFittingEnum.LEFT.value:
        position = True if mouth.x > hand.x else False
    elif side == BikeFittingEnum.RIGHT.value:
        position = True if mouth.x < hand.x else False
    return position


def get_nearmost_wheels_point(circles):
    """Get the nearmost point of the wheels
        args:   
            circles: the circles detected in the image
        returns:
            rightest: the righest point of the left wheel
            leftest: the leftest point of the right wheel
    """
    # get the coordinates of the wheels
    x1, _, _ = circles[0]
    x2, _, _ = circles[1]

    # Check which wheel is the left one
    if x1 < x2:
        left_wheel = circles[0]
        right_wheel = circles[1]
    else:
        left_wheel = circles[1]
        right_wheel = circles[0]

    # get coordinates of the leftest point of the right wheel
    # and the rightest point of the left wheel
    x1, y1, r1 = left_wheel
    x2, y2, r2 = right_wheel

    righest_point = x1 + r1
    leftest_point = x2 - r2

    return (righest_point, y1), (leftest_point, y2)


def print_circle(image, circles):
    """Print the circles in the image
        args:   
            image: the image where the circles will be printed
            circles: the circles to be printed
    """
    output = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles:
            # draw the outer circle
            cv2.circle(output, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    if DEBUG:
        cv2.imshow('Circles', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def print_extended_line(p1, p2, image, color=(0, 0, 255)):
    """
    Extend a line
    args:
        p1: the first point of the line
        p2: the second point of the line distance: the distance to extend the line
    returns:
        new_line: the extended line
    """
    image_width, image_height, _ = image.shape
    if image_width > image_height:
        distance = BikeDetectorEnum.SCALE.value * image_width
    else:
        distance = BikeDetectorEnum.SCALE.value * image_height

    diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    p3_x = int(p1[0] + distance*np.cos(diff))
    p3_y = int(p1[1] + distance*np.sin(diff))
    p4_x = int(p1[0] - distance*np.cos(diff))
    p4_y = int(p1[1] - distance*np.sin(diff))

    cv2.line(image, (p3_x, p3_y), (p4_x, p4_y), color, 2)

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
    """Print the perpendicular line between the wheels center and the center of mass
        args:
            wheels_center: the center of the wheels
            com: the center of mass
            image: the image where the line will be printed
    """
    wheel_a, wheel_b = wheels_center
    x1, y1 = wheel_a
    x2, y2 = wheel_b
    x3, y3 = com
    k = ((y2 - y1) * (x3 - x1) - (x2 - x1) * (y3 - y1)) / ((y2 - y1) ** 2 + (x2 - x1) ** 2)
    x4 = int(x3 - k * (y2 - y1))
    y4 = int(y3 + k * (x2 - x1))
    return (x3, y3), (x4, y4)


def print_point(image, points, text, color=(255, 255, 0)):
    """Print a point in the image
        args:   
            image: the image where the point will be printed
            point: the point to be printed
    """
    for i, point in enumerate(points):
        x, y = point
        cv2.circle(image, (int(x), int(y)), 5, color, -1)
        #cv2.line(image, (int(x), 0), (int(x), image.shape[1]), color, 1)
        cv2.putText(image, text[i], (int(x-10), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(image, text[i], (int(x-10), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if DEBUG:
        cv2.imshow('Point', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def pipeline(image):
    """The pipeline of the project, it will detect the wheels of the bike, 
    and the position of the body, to suggest if the biker has a correct posture.
        args:
            image: the image to be processed
    """
    image_copy = image.copy()
    # 1. Use yolo to detect the bike and restrict the image to the bike
    #       for an easier detection of the wheels.
    crop_coordinates = yolo_object_detection(image)
    start_point, end_point = crop_coordinates
    x, y = start_point
    x_max, y_max = end_point
    cropped_image = image[y:y_max, x:x_max]

    # 2. Detect the wheels
    circles = detect_wheels(cropped_image)
    circles = circles[0, :]

    # 3. Use Mediapipe to detect the landmarks
    # 3.1. Detect the landmarks
    pose_image, landmarks, segmentation_mask = detect_landmarks(image)
    # 3.2. Print the landmarks in the original image
    if DEBUG:
        cv2.imshow('Landmarks', pose_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # 4. Check the circles detected
    # 4.1. Update the x and y coordinates of the circles to match the original image
    if circles is None:
        return None, BikeDetectorEnum.ERROR_NOT_CIRCLE.value, BikeDetectorEnum.RED_COLOR_STR.value
        # raise Exception("Error: No circles detected")

    # 4.2. Check if the center fo the wheels is under the hips
    circles = wheels_under_hips(landmarks, circles)
    if len(circles) != 2:
        return None, BikeDetectorEnum.ERROR_NUM_CIRCLE.value, BikeDetectorEnum.RED_COLOR_STR.value
        # raise Exception("Error: The number of circles detected is not 2")

    # 4.3. update the coordinates of the circles to match the original image
    for circle in circles:
        circle[0] += x
        circle[1] += y
    # 4.4. Print circles in the original image (not the cropped one)
    print_circle(image, circles)

    # 5. Calculate the nearmost point of the two wheels
    left_wheel, right_wheel = get_nearmost_wheels_point(circles)
    # 5.1. Print the nearmost point in the original image
    print_point(image, [left_wheel, right_wheel], ["Left wheel's righest point", "Right wheel's leftest point"])


    # 5. Calculate the center of mass point
    center_of_mass = get_center_of_mass(image, segmentation_mask)

    # 6. Print the line of the center of mass, left wheel and right wheel, that are perpendicular to the
    # line of the wheels
    # 6.1 Get the two circle coordinates
    x1, y1, r1 = circles[0]
    x2, y2, r2 = circles[1]
    circle_center_a = [x1, y1]
    circle_center_b = [x2, y2]

    # 6.2. Compute the perpendicular line between the center of mass and the wheels center
    com_1, com_2 = get_perpendicular_line([circle_center_a, circle_center_b], center_of_mass)
    # 6.3. Compute the parallel line respect to the perpendicular line
    wheel_a_1, wheel_a_2 = get_parallel_line(com_1, com_2, left_wheel)
    wheel_b_1, wheel_b_2 = get_parallel_line(com_1, com_2, right_wheel)

    # 7. Check if the center of mass point is between the two wheels to have a perfect balance , i.e. a good posture
    is_between = check_if_point_is_between_two_points(center_of_mass,
                                                        left_wheel,
                                                        right_wheel)
    if is_between:
        # print("The posture is correct")
        body_suggestion = BikeDetectorEnum.SUGGESTION_CORRECT.value
        body_suggestion_color = BikeDetectorEnum.GREEN_COLOR_STR.value
        color = BikeDetectorEnum.GREEN_COLOR_RGB.value
    else:
        # print("The posture is incorrect")
        body_suggestion = BikeDetectorEnum.SUGGESTION_INCORRECT.value
        body_suggestion_color = BikeDetectorEnum.RED_COLOR_STR.value
        color = BikeDetectorEnum.RED_COLOR_RGB.value

    # 8.1. Extend the line calculated in the point 6, to the image borders
    print_extended_line(com_1, com_2, image, color)
    print_extended_line(wheel_b_1, wheel_b_2, image, (255, 255, 0))
    print_extended_line(wheel_a_1, wheel_a_2, image, (255, 255, 0))

    # 8.2. Print the center of mass point
    print_point(image, [center_of_mass], ["Center of mass"], color)


    # 9. Check if the crank is in the correct position
    crank_position = check_crank_position(image_copy, landmarks, circles[0], circles[1])
    if crank_position:
        crank_suggestion = BikeDetectorEnum.CRANK_SUGGESTION_CORRECT.value
        crank_suggestion_color = BikeDetectorEnum.GREEN_COLOR_STR.value
    else:
        crank_suggestion = BikeDetectorEnum.CRANK_SUGGESTION_INCORRECT.value
        crank_suggestion_color = BikeDetectorEnum.RED_COLOR_STR.value
    
    # 10. Check if the heel is below the foot_index
    heel_position_left, heel_position_right = check_heel_position(landmarks)
    # Check suggestion, if it is True the position is correct, 
    # if it is False the position is incorrect
    # if it is None the position is not detected
    if heel_position_left:
        heel_suggestion_left = BikeDetectorEnum.LEFT_HEEL.value + \
            BikeDetectorEnum.HEEL_SUGGESTION_CORRECT.value
        heel_suggestion_left_color = BikeDetectorEnum.GREEN_COLOR_STR.value
    elif heel_position_left is False:
        heel_suggestion_left = BikeDetectorEnum.LEFT_HEEL.value + \
            BikeDetectorEnum.HEEL_SUGGESTION_INCORRECT.value
        heel_suggestion_left_color = BikeDetectorEnum.RED_COLOR_STR.value
    elif heel_position_left is None:
        heel_suggestion_left = BikeDetectorEnum.LEFT_HEEL.value + \
            BikeDetectorEnum.HEEL_SUGGESTION_UNDETECTED.value
        heel_suggestion_left_color = BikeDetectorEnum.ORANGE_COLOR_STR.value
    # Do the same for the right heel
    if heel_position_right:
        heel_suggestion_right = BikeDetectorEnum.RIGHT_HEEL.value + \
            BikeDetectorEnum.HEEL_SUGGESTION_CORRECT.value
        heel_suggestion_right_color = BikeDetectorEnum.GREEN_COLOR_STR.value
    elif heel_position_right is False:
        heel_suggestion_right = BikeDetectorEnum.RIGHT_HEEL.value + \
            BikeDetectorEnum.HEEL_SUGGESTION_INCORRECT.value
        heel_suggestion_right_color = BikeDetectorEnum.RED_COLOR_STR.value
    elif heel_position_right is None:
        heel_suggestion_right = BikeDetectorEnum.RIGHT_HEEL.value + \
            BikeDetectorEnum.HEEL_SUGGESTION_UNDETECTED.value
        heel_suggestion_right_color = BikeDetectorEnum.ORANGE_COLOR_STR.value

    # 11. Check the distance between the head and the hand, i.e. the handlebar
    head_hand_distance = check_head_hand_distance(landmarks, circles[0], circles[1])
    if head_hand_distance:
        head_hand_suggestion = BikeDetectorEnum.HEAD_HAND_SUGGESTION_CORRECT.value
        head_hand_suggestion_color = BikeDetectorEnum.GREEN_COLOR_STR.value
    else:
        head_hand_suggestion = BikeDetectorEnum.HEAD_HAND_SUGGESTION_INCORRECT.value
        head_hand_suggestion_color = BikeDetectorEnum.RED_COLOR_STR.value
    
    site_suggestion = [body_suggestion,
                        crank_suggestion,
                        heel_suggestion_left,
                        heel_suggestion_right,
                        head_hand_suggestion]

    site_color = [body_suggestion_color,
                    crank_suggestion_color,
                    heel_suggestion_left_color,
                    heel_suggestion_right_color,
                    head_hand_suggestion_color] 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, site_suggestion, site_color

if __name__ == "__main__":
    dir_path = BikeDetectorEnum.TEST_IMAGES_PATH.value
    # list to store files
    images = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            images.append(path)
    
    images_path = [dir_path + image for image in images]

    for image_path in images_path:
        # Read the image
        image = cv2.imread(image_path)
        # Compute the pipeline
        _, suggestion, _ = pipeline(image)
        print(suggestion)