import cv2
from feat import Detector
from feat.utils.image_operations import convert_image_to_tensor


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        return

    detector = Detector(device="cpu")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_tensor = convert_image_to_tensor(frame)
            fex = detector.detect(frame_tensor, data_type="tensor", progress_bar=False)

            print(fex.aus)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
