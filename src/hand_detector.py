import cv2
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        model_path="../models/hand_landmarker.task",
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ):

        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.hand_landmarker = (
            mp.tasks.vision.HandLandmarker
            .create_from_options(options)
        )

    def detect(self, frame):

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert numpy image to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = self.hand_landmarker.detect(mp_image)

        return results

    def get_landmarks(self, results):

        if not results.hand_landmarks:
            return None

        # First detected hand
        hand = results.hand_landmarks[0]

        landmarks = []

        for landmark in hand:

            landmarks.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
            })

        return landmarks

    def close(self):
        self.hand_landmarker.close()