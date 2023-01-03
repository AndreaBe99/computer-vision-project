# This file is used to test the backend functions
import sys
import os
import cv2

sys.path.append("../")

from config import *

from backend.mtb_downhill.bike_detector import pipeline as mtb_downhill_pipeline
from backend.bike_fitting.capture_video_test import pipeline as bike_fitting_pipeline

def test_mtb_downhill():
    # Test pipeline
    # mtb_downhill_pipeline()
    dir_path = BikeDetectorEnum.TEST_IMAGES_PATH.value
    # list to store files
    images = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            images.append(path)

    images_path = [dir_path + image for image in images]

    image = cv2.imread(images_path[2])
    # Compute the pipeline
    _, suggestion, _ = mtb_downhill_pipeline(image)
    return

    for image_path in images_path:
        # Read the image
        image = cv2.imread(image_path)
        # Compute the pipeline
        _, suggestion, _ = mtb_downhill_pipeline(image)
        print(suggestion)
        break

def test_bike_fitting():
    # Test pipeline
    dir_path = ResourcesPath.TEST_VIDEOS_PATH.value
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
        bike_fitting_pipeline(video, 70)
        break


if __name__ == "__main__":
    test_mtb_downhill()
    # test_bike_fitting()


