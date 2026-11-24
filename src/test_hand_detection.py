import cv2

from hand_detector import HandDetector
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = HandDetector()

    print("Hand detection started.")
    print("Press 'q' to quit.")

    while True:

        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame.")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Detect hand
        results = detector.detect(frame)

        # Extract landmarks
        landmarks = detector.get_landmarks(results)

        if landmarks:

            # Draw landmarks manually
            height, width, _ = frame.shape

            for landmark in landmarks:

                x = int(landmark["x"] * width)
                y = int(landmark["y"] * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            connections = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (5, 9),
                (9, 13),
                (13, 17),
            ]

            for start, end in connections:

                x1 = int(landmarks[start]["x"] * width)
                y1 = int(landmarks[start]["y"] * height)

                x2 = int(landmarks[end]["x"] * width)
                y2 = int(landmarks[end]["y"] * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            cv2.putText(
                frame,
                "Hand Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Show wrist coordinates
            wrist = landmarks[0]

            text = (
                f"Wrist: "
                f"X={wrist['x']:.2f} "
                f"Y={wrist['y']:.2f} "
                f"Z={wrist['z']:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "No Hand Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "AI Hand Gesture Control",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()