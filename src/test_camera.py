import cv2
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        # OpenCV uses BGR, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        return results

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        return frame

    def get_landmarks(self, results):
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for landmark in hand.landmark:
            landmarks.append(
                {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                }
            )

        return landmarks