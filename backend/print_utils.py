from config import *
import cv2
import numpy as np
import matplotlib.pyplot as plt


def print_extended_line(p1, p2, image, color=(0, 0, 255)):
    """
    Extend a line
    args:
        p1: the first point of the line
        p2: the second point of the line distance: the distance to extend the line
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


def print_circle(image, circles):
    """
    Print the circles in the image
    args:   
        image: the image where the circles will be printed
        circles: the circles to be printed
    """
    output = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles:
            # draw the outer circle
            cv2.circle(output, (circle[0], circle[1]),
                       circle[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    if DEBUG:
        cv2.imshow('Circles', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def print_point_mtb_descent(image, points, text, color=(255, 255, 0)):
    """
    Print a point in the image
    args:   
        image: the image where the point will be printed
        point: the point to be printed
    """
    for i, point in enumerate(points):
        x, y = point
        cv2.circle(image, (int(x), int(y)), 5, color, -1)
        #cv2.line(image, (int(x), 0), (int(x), image.shape[1]), color, 1)
        cv2.putText(image, text[i], (int(x-10), int(y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, text[i], (int(x-10), int(y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if DEBUG:
        cv2.imshow('Point', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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


def print_point_bike_fitting(points_list, frame, color=(0, 255, 0)):
    """
    Print the points in the list on the frame
    args:
        points_list: list of points
        frame: frame where the points will be printed
        color: color of the points
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


def print_frame(mp_obj, video, frame_index, name: str):
    """Print a frame of a video that is one of the cardinal points with the landmarks
    args:
        mp_obj: MediaPipe object
        video: VideoCapture object
        frame_index: index of the frame
        name: name of the frame, EAST, NORD, WEST, SOUTH
    """
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = get_frame_from_point(video, frame_index)
    frame, landmark, _ = mp_obj.detect_image_landmarks(frame)
    if DEBUG:
        cv2.imshow('FRAME '+name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return landmark

def print_logs(logs):
    """Print the logs in the console
    args:
        logs: list of logs
    """
    print("@"*50)
    for log in logs:
        print(log)
    print("@"*50)
