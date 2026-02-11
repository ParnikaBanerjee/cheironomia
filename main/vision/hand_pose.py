# vision/hand_pose.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vision.smoothing import ExponentialSmoother


class HandPoseTracker:
    def __init__(self):
        # -------- HAND LANDMARKER --------
        hand_base_options = python.BaseOptions(
            model_asset_path="models/hand_landmarker.task"
        )

        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2
        )

        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        # -------- POSE LANDMARKER --------
        pose_base_options = python.BaseOptions(
            model_asset_path="models/pose_landmarker_full.task"
        )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options
        )

        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        self.smoother = ExponentialSmoother(alpha=0.6)

    def process(self, rgb_frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        hand_result = self.hand_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)

        arms = self.extract_arms(pose_result.pose_landmarks)

        data = {
            "hands": hand_result.hand_landmarks if hand_result.hand_landmarks else [],
            "pose": pose_result.pose_landmarks if pose_result.pose_landmarks else [],
            "arms": arms
        }

        return data

    def extract_arms(self, pose_landmarks):
        if not pose_landmarks:
            return None

        lm = pose_landmarks[0]

        return {
            "left_shoulder": self.smoother.smooth(
                "left_shoulder", (lm[11].x, lm[11].y, lm[11].z)
            ),
            "left_elbow": self.smoother.smooth(
                "left_elbow", (lm[13].x, lm[13].y, lm[13].z)
            ),
            "left_wrist": self.smoother.smooth(
                "left_wrist", (lm[15].x, lm[15].y, lm[15].z)
            ),

            "right_shoulder": self.smoother.smooth(
                "right_shoulder", (lm[12].x, lm[12].y, lm[12].z)
            ),
            "right_elbow": self.smoother.smooth(
                "right_elbow", (lm[14].x, lm[14].y, lm[14].z)
            ),
            "right_wrist": self.smoother.smooth(
                "right_wrist", (lm[16].x, lm[16].y, lm[16].z)
            ),
        }
