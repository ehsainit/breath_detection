import cv2
import pyrealsense2 as rs

from src.face_detector import Detector

if __name__ == '__main__':
    # create face detector instance
    face_detector = Detector()

    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)

    if cap.isOpened():  # try to get the first frame
        rval, frame = cap.read()
    else:
        rval = False

    while rval:
        results, inference_time = face_detector.detect(frame)

        # display fps
        cv2.putText(frame, f'{round(1 / inference_time, 2)} FPS', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)

        if results:
            x, y, w, h = results
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # text = "%s (%s)" % (name, round(confidence, 2))
            # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #            0.5, color, 2)

        cv2.imshow("preview", frame)

        rval, frame = cap.read()

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    cap.release()