import mediapipe as mp
import cv2
import ffmpeg
from config import *

class MediaPipePoseEstimation:
    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    # mp_pose = mp.solutions.pose

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
    
    def detect_image_landmarks(self, image):
        """
        Detect the landmarks of the body
            args:   
                image: the image where the landmarks will be detected
            returns:
                landmarks: the landmarks of the body
        """
        # Determine width and height of the image
        with self.mp_pose.Pose(
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

            self.mp_drawing.draw_landmarks(
                image_copy,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())

            if DEBUG:
                cv2.imshow('MediaPipe Pose', results.segmentation_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return image_copy, landmarks, results.segmentation_mask
    
    def detect_video_landmarks(self, video):
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
        result = cv2.VideoWriter(
            BikeFittingEnum.VIDEO_LANDMARKS_PATH_AVI.value, fourcc, 25, size)

        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            i = 0
            while i < int(video.get(cv2.CAP_PROP_FRAME_COUNT)):
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
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

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
        ffmpeg.input(BikeFittingEnum.VIDEO_LANDMARKS_PATH_AVI.value).output(
            BikeFittingEnum.VIDEO_LANDMARKS_PATH_MP4.value).run()

        cv2.destroyAllWindows()
        return landmark_list, frame

    def get_landmarks_from_frame(self, frame):
        """
        Get the landmarks from a frame, and print them
        args:
            frame: frame to process
        return:
            image: frame with the landmarks
            landmarks: list of landmarks
        """
        with self.mp_pose.Pose(
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
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image, results.pose_landmarks.landmark
