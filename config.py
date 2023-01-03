from enum import Enum
from os import environ as env
import multiprocessing

# Flask
PORT = int(env.get("PORT", 8080))
DEBUG_MODE = int(env.get("DEBUG_MODE", 1))

# Gunicorn config
bind = ":" + str(PORT)
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2 * multiprocessing.cpu_count()

# Google Cloud Storage
CLOUD_STORAGE_BUCKET = "cv-project-bucket"
GOOGLE_APPLICATION_CREDENTIALS = "GAC.json"

# Backend
DEBUG = False

class AnglesRange(Enum):
    # Knee angle is between 90° and 190°, with 140° as the ideal value,
    KNEE = (90, 140, 190)
    # Ankle angle is between 80° and 180°, with 120° as the ideal value
    ANKLE = (80, 120, 180)
    # Torso angle is between 40° and 140°, with 90° as the ideal value
    TORSO = (40, 90, 140)

class ResourcesPath(Enum):
    # Backend path
    BACKEND_PATH = 'backend/mtb_downhill/'
    UPLOADED_PHOTOS_DEST = 'static/uploads/images/'
    UPLOADED_VIDEOS_DEST = 'static/uploads/videos/'
    DIRECTION_ICON = "../static/images/"

    # bike_detector.py
    TEST_IMAGES_PATH = "backend/resources/downhill_posture/"

    # yolo_object_detection.py
    TEST_VIDEOS_PATH = "backend/resources/bike_fitting/videos/"
    WEIGHTS = "backend/mtb_downhill/yolo_object_detection/yolov3.weights"
    CONFIG = "backend/mtb_downhill/yolo_object_detection/yolov3.cfg"

    # bike_fitting.py
    VIDEO_NAME= "video_landmarks"
    VIDEO_FORMAT_AVI = ".avi"
    VIDEO_FORMAT_MP4 = ".mp4"
    VIDEO_FORMAT_WEBM = ".webm"

    PHOTO_NAME = "pedaling_progress"
    PHOTO_FORMAT = ".png"

    


class FrontendEnum(Enum):
    SECRET_KEY = 'abcdefg123456'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    BACKEND_PATH = ResourcesPath.BACKEND_PATH.value
    UPLOADED_PHOTOS_DEST = ResourcesPath.UPLOADED_PHOTOS_DEST.value
    UPLOADED_VIDEOS_DEST = ResourcesPath.UPLOADED_VIDEOS_DEST.value
    VIDEOS = ['mp4', 'mov', 'avi', 'flv', 'wmv', 'mkv']
    SADDLE_STR = "Move the SADDLE "
    HANDLEBAR_STR = "Move the HANDLEBAR "
    UP_STR = "UP of "
    DOWN_STR = "DOWN of "
    BACKWARD_STR = "BACKWARD of "
    FORWARD_STR = "FORWARD of "
    DIRECTION_ICON = ResourcesPath.DIRECTION_ICON.value
    ICON_DOWN = DIRECTION_ICON + "arrow-down.png"
    ICON_UP = DIRECTION_ICON + "arrow-up.png"
    ICON_LEFT = DIRECTION_ICON + "arrow-left.png"
    ICON_RIGHT = DIRECTION_ICON + "arrow-right.png"

    SADDLE_OK_STR = "The SADDLE is in the correct position."
    SADDLE_OK_HEIGHT_STR = "The SADDLE height is correct."

    HANDLEBAR_OK_STR = "The HANDLEBAR is in the correct position."
    ICON_OK = ResourcesPath.DIRECTION_ICON.value + "check-mark.png"

class YoloDetectorEnum(Enum):
    # TEST_VIDEOS_PATH = ResourcesPath.TEST_VIDEOS_PATH
    # download weights: wget https://pjreddie.com/media/files/yolov3.weights
    WEIGHTS = ResourcesPath.WEIGHTS.value
    # wget "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    CONFIG = ResourcesPath.CONFIG.value
    CLASSES = ["person", "bike"]

class BikeDetectorEnum(Enum):
    TEST_IMAGES_PATH = ResourcesPath.TEST_IMAGES_PATH.value
    SCALE = 10
    ERROR_NOT_CIRCLE = "Error: No circles detected"
    ERROR_NUM_CIRCLE = "Error: Sorry!!! The number of circles detected is not 2"
    RED_COLOR_STR = "red"
    RED_COLOR_RGB = (0, 0, 255)
    GREEN_COLOR_STR = "green"
    GREEN_COLOR_RGB = (0, 255, 0)
    ORANGE_COLOR_STR = "orange"
    SUGGESTION_CORRECT = "Your UPPER BODY posture is correct, i.e. your center of mass is within the proper horizontal area."
    SUGGESTION_INCORRECT = "Your UPPER BODY posture is incorrect, i.e. your center of mass is not within the proper horizontal area."

    CRANK_SUGGESTION_CORRECT = "The CRANKS are in the correct position i.e. they are placed on the same horizontal line parallel to the ground."
    CRANK_SUGGESTION_INCORRECT = "The CRANKS are in the incorrect position i.e. they are not placed on the same horizontal line parallel to the ground."

    RIGHT_HEEL = "The RIGHT HEEL is "
    LEFT_HEEL = "The LEFT HEEL is "
    HEEL_SUGGESTION_CORRECT = "in the correct position i.e. it is under the crank axis."
    HEEL_SUGGESTION_INCORRECT = "in an incorrect position, i.e. it is above the crank axis."
    HEEL_SUGGESTION_UNDETECTED = "not detected."

    HEAD_HAND_SUGGESTION_CORRECT = "The HEAD and HANDS are in the correct position, i.e. the head is behind the arms for better stability on the rear wheel."
    HEAD_HAND_SUGGESTION_INCORRECT = "The HEAD and HANDS are in an incorrect position, i.e. the head is not behind the arms for better stability on the rear wheel."

class BikeFittingEnum(Enum):
    VIDEO_LANDMARKS_PATH_AVI = ResourcesPath.UPLOADED_VIDEOS_DEST.value +\
                                ResourcesPath.VIDEO_NAME.value+\
                                ResourcesPath.VIDEO_FORMAT_AVI.value

    VIDEO_LANDMARKS_PATH_MP4 = ResourcesPath.UPLOADED_VIDEOS_DEST.value +\
                                ResourcesPath.VIDEO_NAME.value+\
                                ResourcesPath.VIDEO_FORMAT_MP4.value

    VIDEO_LANDMARKS_PATH_WEBM = ResourcesPath.UPLOADED_VIDEOS_DEST.value +\
                                ResourcesPath.VIDEO_NAME.value+\
                                ResourcesPath.VIDEO_FORMAT_WEBM.value
    
    

    PHOTO_PEDALING_PROGRESS_PATH = ResourcesPath.UPLOADED_PHOTOS_DEST.value +\
                                ResourcesPath.PHOTO_NAME.value+\
                                ResourcesPath.PHOTO_FORMAT.value
    LEFT = "left"
    RIGHT = "right"