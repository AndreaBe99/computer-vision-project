import mediapipe as mp
import cv2
import ffmpeg
from config import *

class MediaPipePoseEstimation:

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
    def draw_landmarks(self, image, results):
        """
        Draw the landmarks on the image
        args:
            image: the image where the landmarks will be detected
            results: the results of the landmarks detection
        """
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    def compute_image(self, image, pose):
        """
        Compute the landmarks of the image
        args:
            image: the image where the landmarks will be detected
            pose: the pose object
        returns:    
            image: the image with the landmarks printed
            results: the results of the landmarks detection
        """
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if not results.pose_landmarks:
            return False
        image.flags.writeable = True
        self.draw_landmarks(image, results)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def detect_image_landmarks(self, image):
        """
        Detect the landmarks of the body
            args:   
                image: the image where the landmarks will be detected
            returns:
                img: the image with the landmarks printed
                landmarks: the landmarks of the body
                seg_mask: the segmentation mask of the human body
        """
        # Determine width and height of the image
        with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5) as pose:

            img = image.copy()
            img, results = self.compute_image(img, pose)

            if DEBUG:
                cv2.imshow('MediaPipe Pose', results.segmentation_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            landmarks = results.pose_landmarks.landmark
            seg_mask = results.segmentation_mask

            return img, landmarks, seg_mask
    
    def detect_video_landmarks(self, video):
        """
        Compute the landmarks of the video and save them in a list,
        and save the video with the landmarks in a file (convert it for the browser)
        args:
            video: VideoCapture object
        return:
            landmarks_list: list of landmarks, one list for each frame, and each landmark is a list of 33 elements
            image: last frame of the video, used to print the tests
        """
        # We need to set resolutions so, convert them from float to integer.
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        size = (frame_width, frame_height)
        # Define the codec and create VideoWriter object.
        # The output is stored in '.avi' file.
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        result = cv2.VideoWriter(BikeFittingEnum.VIDEO_LANDMARKS_PATH_AVI.value, fourcc, 25, size)

        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            i = 0
            landmark_list = []
            while i < int(video.get(cv2.CAP_PROP_FRAME_COUNT)):
                success, image = video.read()
                i += 1
                if not success:
                    break
                image, results = self.compute_image(image, pose)
                landmark_list.append(results.pose_landmarks.landmark)

                if DEBUG:
                    cv2.imshow('MediaPipe Pose', image)
                    # Quit with q
                    if cv2.waitKey(1) == ord('q'):
                        break

                # Write the frame into the video file
                result.write(image)
        result.release()
        # Opencv doesn't support HEVC codec, and other codecs are
        # not supported by the browser, so we need to convert the
        # video from AVI to MP4 with ffmpeg a wrapper for ffmpeg cli
        ffmpeg.input(BikeFittingEnum.VIDEO_LANDMARKS_PATH_AVI.value).output(
            BikeFittingEnum.VIDEO_LANDMARKS_PATH_MP4.value).run()
        cv2.destroyAllWindows()
        return landmark_list, image