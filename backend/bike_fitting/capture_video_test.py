import cv2
import os
import math
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

import subprocess as sp
import shlex

# import imageio.v2 as iio
import ffmpeg


from config import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


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

def codec_video(width, height, fps, output_filename):
    # Open ffmpeg application as sub-process
    # FFmpeg input PIPE: RAW images in BGR color format
    # FFmpeg output MP4 file encoded with HEVC codec.
    # Arguments list:
    # -y                   Overwrite output file without asking
    # -s {width}x{height}  Input resolution width x height (1344x756)
    # -pixel_format bgr24  Input frame color format is BGR with 8 bits per color component
    # -f rawvideo          Input format: raw video
    # -r {fps}             Frame rate: fps (25fps)
    # -i pipe:             ffmpeg input is a PIPE
    # -vcodec libx265      Video codec: H.265 (HEVC)
    # -pix_fmt yuv420p     Output video color space YUV420 (saving space compared to YUV444)
    # -crf 24              Constant quality encoding (lower value for higher quality and larger output file).
    # {output_filename}    Output file name: output_filename (output.mp4)
    process = sp.Popen(shlex.split(
        f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {output_filename}'),
                       stdin=sp.PIPE)
    
    return process


def get_landmarks(video):
    """
    Compute the landmarks of the video and save them in a list,
    and save the video with the landmarks in a file (convert it for the browser)
    args:
        video: VideoCapture object
    return:
        landmarks_list: list of landmarks, one list for each frame, and each landmark is a list of 33 elements
        frame: last frame of the video, used to print the tests
    """
    landmark_list = []
    frame = None

    # We need to set resolutions so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)


    # Define the codec and create VideoWriter object.
    # The output is stored in '.avi' file.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    result = cv2.VideoWriter(BikeFittingEnum.VIDEO_LANDMARKS_PATH_AVI.value, fourcc, 25, size)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        i = 0
        while i<int(video.get(cv2.CAP_PROP_FRAME_COUNT)):
            success, image = video.read()
            i += 1
            if not success:
                break
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            frame = image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            landmarks = results.pose_landmarks.landmark
            # Append the landmarks of each frame to the landmark_list
            landmark_list.append(landmarks)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if DEBUG:
                cv2.imshow('MediaPipe Pose', image)

            # Write the frame into the video file
            result.write(image)

            # Quit with q
            if cv2.waitKey(1) == ord('q'):
                break

    result.release()
    
    # Opencv doesn't support HEVC codec, and other codecs are 
    # not supported by the browser, so we need to convert the 
    # video from AVI to MP4 with ffmpeg a wrapper for ffmpeg cli
    ffmpeg.input(BikeFittingEnum.VIDEO_LANDMARKS_PATH_AVI.value).output(BikeFittingEnum.VIDEO_LANDMARKS_PATH_MP4.value).run()

    cv2.destroyAllWindows()
    return landmark_list, frame

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

def save_plot(point_list):
    foot_position = [point[1] for point in point_list]
    frames = [i for i in range(len(point_list))]
    plt.figure(figsize=(20, 5))
    plt.plot(frames, foot_position)
    plt.title('Pedaling progress')
    plt.xlabel('Frames')
    plt.ylabel('Height')    
    plt.savefig('static/uploads/images/pedaling_progress.png')
    # plt.show()

def get_pedals_cardinal_point(image, landmarks_list: list, side: int):
    """
    Compute the centroids of the middle point between foot and hill, for all frames, then obtain the cardinal points
    using the length of the rider's foot
    args:
        image: image of the video
        landmarks_list: list of landmarks of all frames
    return:
        cardinal_points: list of cardinal points, [point_east, point_south, point_west, point_nord]
    """
    image_height, image_width, _ = image.shape
    foot_list = []
    for landmarks in landmarks_list:
        foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX + side]
        heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL + side]
        # compute middle point between foot and heel
        point = [foot.x + (heel.x - foot.x) / 2, foot.y + (heel.y - foot.y) / 2]
        ### PRINT TEST ###
        cv2.circle(image, (int(point[0] * image_width), int(point[1]*image_height)), 1, (0, 0, 255), 3)
        ##################
        foot_list.append([point[0], point[1]])


    # Plot the points and save the image
    save_plot(foot_list)

    # Compute the centroid
    centroid = get_centroid(foot_list)
    ### PRINT TEST ###
    cv2.circle(image, (int(centroid[0] * image_width), int(centroid[1]*image_height)), 2, (0, 255, 255), 3)
    ##################

    # Compute distance between foot and  heel
    # distance = get_distance([foot.x, foot.y], [heel.x, heel.y]) / 2
    # add 40% to the distance
    # distance += distance * 0.4

    # Compute the four cardinal point around centroid
    ### TEST ###
    max_x = max([p[0] for p in foot_list])
    min_x = min([p[0] for p in foot_list])
    max_y = max([p[1] for p in foot_list])
    min_y = min([p[1] for p in foot_list])
    point_east = [int(max_x * image_width), int(centroid[1] * image_height)]
    point_south = [int(centroid[0] * image_width), int(max_y * image_height)]
    point_west = [int(min_x * image_width), int( centroid[1] * image_height)]
    point_north = [int(centroid[0] * image_width), int(min_y * image_height)]

    # point_east = [int( (centroid[0] + distance) * image_width), int( centroid[1] * image_height)]
    # point_south = [int( centroid[0] * image_width), int( (centroid[1] + distance) * image_height)]
    # point_west = [int( (centroid[0] - distance) * image_width), int( centroid[1] * image_height)]
    # point_north = [int( centroid[0] * image_width), int( (centroid[1] - distance) * image_height)]
    ############

    ### PRINT TEST ###
    cv2.circle(image, (int(point_east[0]), int(point_east[1])), 5, (255, 255, 0), 3)
    cv2.circle(image, (int(point_south[0]), int(point_south[1])), 5, (255, 255, 0), 3)
    cv2.circle(image, (int(point_west[0]), int(point_west[1])), 5, (255, 255, 0), 3)
    cv2.circle(image, (int(point_north[0]), int(point_north[1])), 5, (255, 255, 0), 3)
    ##################

    return [point_east, point_south, point_west, point_north]


def get_cardinal_points_list(image, landmarks_list: list, cardinal_points: list, side: int):
    """Compute the frames that are at hours 3, 6, 9, 12 of the pedal"""
    image_height, image_width, _ = image.shape
    # Compute a list of the points near to the respetive cardinal point, list: [point, frame_index]
    point_list_east, point_list_south, point_list_west, point_list_north = [], [], [], []

    # Set a threshold to collect points, use the mean vertical and horizontal distance between the points
    horizontal_distance = get_distance(cardinal_points[0], cardinal_points[2])
    vertical_distance = get_distance(cardinal_points[1], cardinal_points[3])
    threshold = ((horizontal_distance + vertical_distance)/2)
    # Length of each list when we divide mean by:
    #   - 5:    25  23  17  22
    #   - 7.5:  15  6   7   12
    #   - 10:   8   3   2   5
    threshold = threshold / 7.5

    point_east, point_south, point_west, point_north = cardinal_points
    # Loop over the frame
    for i, landmarks in enumerate(landmarks_list):
        foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX + side]
        heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL + side]
        # compute middle point between foot and heel
        point = [int( (foot.x + (heel.x - foot.x) / 2) * image_width),
                 int( (foot.y + (heel.y - foot.y) / 2) * image_height)]

        # For each hour get all coordinates frames that are in the threshold
        if get_distance(point, point_east) <= threshold:
            point_list_east.append([[point[0], point[1]], i])
        elif get_distance(point, point_south) <= threshold:
            point_list_south.append([[point[0], point[1]], i])
        elif get_distance(point, point_west) <= threshold:
            point_list_west.append([[point[0], point[1]], i])
        elif get_distance(point, point_north) <= threshold:
            point_list_north.append([[point[0], point[1]], i])

    return [point_list_east, point_list_south, point_list_west, point_list_north]


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

def get_knee_angle(landmarks: list, side: int):
    """
    Compute the angle between the ankle, knee and the hip
    args:
        landmarks: list of landmarks of a frame
    return:
        maximum_angle: maximum angle between ankle, knee and hip
    """
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE + side]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE + side]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP + side]
    return get_angle([ankle.x, ankle.y], [knee.x, knee.y], [hip.x, hip.y])

def get_ankle_angle(landmarks: list, side: int):
    """
    Compute the angle between the foot, ankle and the knee
    args:
        landmarks: list of landmarks of a frame
    return:
        maximum_angle: maximum angle between foot, ankle and the knee
    """
    foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX + side]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE + side]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE + side]
    return get_angle([foot.x, foot.y], [ankle.x, ankle.y], [knee.x, knee.y])

def get_mean_angles_saddle_height(points_list: list, landmarks: list, side: int):
    """
    Compute the angles of the knee and the ankle, for each point in the list, then compute the mean
    args:
        points_list: list of points
        landmarks: list of landmarks
    return:
        mean_knee_angle: mean of the knee angles
        mean_ankle_angle: mean of the ankle angles
    """
    knee_angles_sum = 0
    ankle_angles_sum = 0
    if len(points_list) < 1:
        raise Exception('The list of points is empty')
    for point in points_list:
        knee_angles_sum += get_knee_angle(landmarks[point[1]], side)
        ankle_angles_sum += get_ankle_angle(landmarks[point[1]], side)
    knee_angles_mean = knee_angles_sum / len(points_list)
    ankle_angles_mean = ankle_angles_sum / len(points_list)
    return knee_angles_mean, ankle_angles_mean

def get_mean_distance_saddle_foreaft(points_list, landmarks, side):
    """
    Compute for each frame in the list the horizontal distance between the foot and the knee, then compute the mean
    args:
        points_list: list of points
        landmarks: list of landmarks
    return:
        mean_distance: mean of the distances
    """
    distance_sum = 0
    if len(points_list) < 1:
        raise Exception('The list of points is empty')
    # If the side is right, the distance is positive, otherwise is negative
    for point in points_list:
        foot = landmarks[point[1]][mp_pose.PoseLandmark.LEFT_FOOT_INDEX + side]
        knee = landmarks[point[1]][mp_pose.PoseLandmark.LEFT_KNEE + side]
        distance_sum += knee.x - foot.x
    distance_mean = distance_sum / len(points_list)
    return distance_mean

def get_mean_torso_angle(landmark_list, side):
    """
    Compute the angle between the hips, the shoulders and the elbows for each frame, then compute the mean
    args:
        landmark_list: list of landmarks
    return:
        mean_angle: mean of the angles
    """
    angle_sum = 0
    for landmarks in landmark_list:
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP + side]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER + side]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW + side]
        angle_sum += get_angle([hip.x, hip.y], [shoulder.x, shoulder.y], [elbow.x, elbow.y])
    angle_mean = angle_sum / len(landmark_list)
    return angle_mean


######### TEST #########
def print_point(points_list, frame, color=(0, 255, 0)):
    """
    Print the points in the list on the frame
    """
    for point in points_list:
        cv2.circle(frame, (int(point[0][0]), int(point[0][1])), 3, color, -1)

def get_frame_from_point(video, frame_index):
    """
    Loop through the video and return the frame at the given index
    args:
        video: VideoCapture object
        frame_index: index of the frame
    return:
        frame: frame at the given index
    """
    image = None
    i = 0
    # while video.isOpened() and i < frame_index:
    while i < frame_index:
        success, image = video.read()
        i += 1
        if not success:
            break
    return image

def get_landmarks_from_frame(frame):
    """
    Get the landmarks from a frame, and print them
    args:
        frame: frame to process
    return:
        image: frame with the landmarks
        landmarks: list of landmarks
    """
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        if not results.pose_landmarks:
            return False
        image.flags.writeable = True
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results.pose_landmarks.landmark

def print_frame(video, frame_index, name: str):
    """Print a frame of a video that is one of the cardinal points with the landmarks
    args:
        video: VideoCapture object
        frame_index: index of the frame
        name: name of the frame, EAST, NORD, WEST, SOUTH
    """
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = get_frame_from_point(video, frame_index)
    frame, landmark = get_landmarks_from_frame(frame)
    if DEBUG:
        cv2.imshow('FRAME '+name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return landmark
########################


def get_length_third_side(a, b, alpha):
    """
    Given the length of two sides and the angle between them, compute the length of the third side
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
    # Now print with printf anf two decimal places
    if DEBUG:
        print('a:%.2fcm  b:%.2fcm  c:%.2fcm  alpha:%.2f°  beta:%.2f°  gamma:%.2f°' % (a, b, c, alpha, beta, gamma))
    return c

def get_saddle_distance(landmarks, side, total_length, angle):
    """
    Given the rider's length up to the hip, find the length of the ankle-knee and knee-hip
    args:
        landmarks: list of landmarks
        side: side of the rider
        total_length: length of the rider up to the hip in cm
    return:
        distance_real_world: distance in real world between foot and hip in cm
        ratio: ratio between the distance in real word and the image
    """
    ankle_point = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE + side]
    knee_point = landmarks[mp_pose.PoseLandmark.LEFT_KNEE + side]
    hip_point = landmarks[mp_pose.PoseLandmark.LEFT_HIP + side]

    distance_knee_hip = get_distance([knee_point.x, knee_point.y],
                                     [hip_point.x, hip_point.y])
    distance_ankle_knee = get_distance([ankle_point.x, ankle_point.y],
                                       [knee_point.x, knee_point.y])
    distance_ankle_hip = get_distance([ankle_point.x, ankle_point.y],
                                  [hip_point.x, hip_point.y])

    # Compute the distance in real world of the knee and the ankle
    # total_length : knee_hip = distance_knee_hip + distance_ankle_knee : distance_knee_hip
    # total_length : ankle_knee = distance_knee_hip + distance_ankle_knee : distance_ankle_knee
    knee_hip = total_length * distance_knee_hip / (distance_knee_hip + distance_ankle_knee)
    ankle_knee = total_length * distance_ankle_knee / (distance_knee_hip + distance_ankle_knee)

    # Compute the distance between the foot and the hip in real world
    # We can't use the ratio and 'distance_ankle_hip', because it always refers to the image case,
    # we need to compute also the distance in the general case without the image reference
    distance_real_world = get_length_third_side(ankle_knee, knee_hip, angle)

    # Compute the ratio between the two distances
    ratio =  knee_hip / distance_knee_hip

    if DEBUG:
        print("@" * 100)
        print('Image Distance  - knee-hip: %.2f' % distance_knee_hip)
        print('Image Distance - ankle_knee: %.2f' % distance_ankle_knee)
        print('Image Distance - ankle-hip: %.2f' % distance_ankle_hip)
        print('Real World Distance - knee-hip: %.2fcm' % knee_hip)
        print('Real World Distance - ankle-knee: %.2fcm' % ankle_knee)
        print('Real World Distance - ankle-hip: %.2fcm' % distance_real_world)
        print('Real World/Image Ratio: %.2f' % ratio)
        print("@" * 100)

    return distance_real_world, ratio

def get_handlebar_distance(landmarks, ratio, torso_angle, side):
    """
    Compute the handlebar distance in real world, given the radio between image and real world.
    args:
        landmarks: list of landmarks
        ratio: ratio between image and real world
        torso_angles: list of angles of the torso
        side: side of the rider
    return:
        handlebar_distance: distance between the handlebar and the hip in real world
    """
    distance_hip_shoulder = 0
    distance_shoulder_wrist = 0
    distance_hip_wrist = 0
    for landmark in landmarks:
        hip_point = landmark[mp_pose.PoseLandmark.LEFT_HIP + side]
        shoulder_point = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER + side]
        elbow_point = landmark[mp_pose.PoseLandmark.LEFT_ELBOW + side]

        distance_hip_shoulder += get_distance([hip_point.x, hip_point.y], [shoulder_point.x, shoulder_point.y])
        distance_shoulder_wrist += get_distance([shoulder_point.x, shoulder_point.y], [elbow_point.x, elbow_point.y])
        distance_hip_wrist += get_distance([hip_point.x, hip_point.y], [elbow_point.x, elbow_point.y])

    mean_distance_hip_shoulder  = distance_hip_shoulder / len(landmarks)
    mean_distance_shoulder_elbow = distance_shoulder_wrist / len(landmarks)
    mean_distance_hip_wrist = distance_hip_wrist / len(landmarks)

    # Multiply by the ratio to get the real world distance
    mean_distance_hip_shoulder *= ratio
    mean_distance_shoulder_elbow *= ratio
    mean_distance_hip_wrist *= ratio


    # Compute the distance between the handlebar and the hip in real world
    # We can't use the 'mean_distance_hip_wrist' because it always refers to the image case, we need to compute also
    # the distance in the general case without the image reference
    handlebar_distance = get_length_third_side(mean_distance_hip_shoulder, mean_distance_shoulder_elbow, torso_angle)

    if DEBUG:
        print("@" * 100)
        print('Handlebar distance: %.2fcm' % handlebar_distance)
        print("@" * 100)
    return handlebar_distance

def pipeline(video, person_height:int):
    """
    Pipeline of the algorithm
    args:
        video: VideoCapture object
    return:
        knee_angles_mean_south: mean of the knee angles for the south points
        ankle_angles_mean_south: mean of the ankle angles for the south points
        knee_angles_mean_north: mean of the knee angles for the north points
        ankle_angles_mean_north: mean of the ankle angles for the north points
        distance_mean: mean of the horizontal distance between the foot and the knee
        torso_angles_mean: mean of the angle between the shoulders and the hips, and a point on the same x axes of the hip

    """
    # Compute the landmarks for each frame and store it in a list, and return the last frame to display the tests
    landmark_list, frame = get_landmarks(video)

    # If the list is empty, return None
    # This means that the video is not valid
    if len(landmark_list) == 0:
        return None, None, None, None, None, None

    # Get index of middle frame
    middle_frame = int(len(landmark_list) / 2)
    # Use the middle frame to get the bike direction
    bike_direction = get_bike_direction(landmark_list[middle_frame])
    side = 0 if bike_direction == BikeFittingEnum.LEFT.value else 1

    # Compute the centroid of the landmarks of the middle point between the foot and the heel, for each frame
    # Use it to compute the four cardinal points
    cardinal_points = get_pedals_cardinal_point(frame, landmark_list, side)
    # point_east, point_south, point_west, point_north = cardinal_points

    # Get all the points that are around the four cardinal point in the threshold
    # We consider the threshold as the radius of a circle, if a point is inside the circle it is considered
    cardinal_points_list = get_cardinal_points_list(frame, landmark_list, cardinal_points, side)
    point_list_east, point_list_south, point_list_west, point_list_north = cardinal_points_list

    ######### PRINT TEST #########
    if DEBUG:
        print("@"*100)
        print("EAST", len(point_list_east))
        print("SOUTH", len(point_list_south))
        print("WEST", len(point_list_west))
        print("NORTH", len(point_list_north))
        print("@" * 100)

    # Frame with all cardinal point founded
    print_point(point_list_east, frame)
    print_point(point_list_south, frame)
    print_point(point_list_west, frame)
    print_point(point_list_north, frame)

    # if DEBUG:
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey(0)
    #    cv2.destroyAllWindows()

    # One frame for each cardinal point with the landmarks
    landmarks_east = print_frame(video, point_list_east[0][1], "EAST")
    landmarks_south = print_frame(video, point_list_south[0][1], "SOUTH")
    landmarks_west = print_frame(video, point_list_west[0][1], "WEST")
    landmarks_north = print_frame(video, point_list_north[0][1], "NORTH")
    video.release()
    ################################

    # For each list of the cardinal point we compute the respective angle for the position
    # REFERENCE FOR TASK:
    #   - 1: https://bikedynamics.co.uk/fit02.htm
    #   - 2: https://bikedynamics.co.uk/saddleforeaft.htm
    #   - 3: https://bikedynamics.co.uk/fit03.htm

    # 1 SADDLE HEIGHT: To suggest the saddle height, we need the coordinates at south and north
    # 1.1 Compute the angle for south points, i.e. when the leg is completely extended
    knee_angles_mean_south, ankle_angles_mean_south = get_mean_angles_saddle_height(point_list_south, landmark_list, side)
    # 1.2 Compute the angle for north points, i.e. when the leg is completely flexed
    knee_angles_mean_north, ankle_angles_mean_north = get_mean_angles_saddle_height(point_list_north, landmark_list, side)

    # 2 SADDLE FORE/AFT: To suggest the saddle fore/aft position, we need the coordinates at east and west
    # 2.1 Compute the horizontal distance between the foot and the knee, for the pedal in east direction
    # when bike is on 'right' side or in west direction when bike is on 'left' side
    # If left side consider east direction, if right side consider west direction
    if side == 0:
        distance_mean = get_mean_distance_saddle_foreaft(point_list_west, landmark_list, side)
    elif side == 1:
        distance_mean = get_mean_distance_saddle_foreaft(point_list_east, landmark_list, side)

    # 3 HANDLEBAR REACH: To suggest the handlebar reach, we need to compute the torso angle
    torso_angles_mean = get_mean_torso_angle(landmark_list, side)

    if DEBUG:
        # Print the results
        print("@"*100)
        print("knee_angles_mean_south, 141°: ", knee_angles_mean_south)
        print("ankle_angles_mean_south, 130°: ", ankle_angles_mean_south)
        print("knee_angles_mean_north, 73°: ", knee_angles_mean_north)
        print("ankle_angles_mean_north, 114°: ", ankle_angles_mean_north)
        print("distance_mean: ", distance_mean)
        print("40° < torso_angles_mean < 140°: ", torso_angles_mean)
        print("@" * 100)

    # Correct angles is 141°
    correct_saddle_distance, ratio = get_saddle_distance(
        landmarks_south, side, person_height, AnglesRange.KNEE.value[1])
    measured_saddle_distance, _ = get_saddle_distance(
        landmarks_south, side, person_height, knee_angles_mean_south)

    # Compute how much move up or down the saddle
    move_saddle_up_down = correct_saddle_distance - measured_saddle_distance

    # Compute how much move forward or backward the saddle
    move_saddle_forward_backward = distance_mean * ratio
    
    # Compute how much move the handlebar
    correct_handlebar_distance = get_handlebar_distance(
        landmark_list, ratio, AnglesRange.TORSO.value[1], side)
    measured_handlebar_distance = get_handlebar_distance(landmark_list, ratio, torso_angles_mean, side)
    move_handlebar_forward_backward = correct_handlebar_distance - measured_handlebar_distance

    return [move_saddle_up_down, move_saddle_forward_backward, move_handlebar_forward_backward, side], [knee_angles_mean_south, ankle_angles_mean_south, torso_angles_mean]

if __name__ == "__main__":
    dir_path = "../resources/bike_fitting/videos/"
    # list to store files
    videos = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            videos.append(path)
    videos_path = [dir_path + video for video in videos]

    for video_path in videos_path:
        # We retrieve the landmarks of each frame and a single frame to display the tests
        video = cv2.VideoCapture(video_path)
        pipeline(video, 70)
        break