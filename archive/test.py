import cv2

if __name__ == '__main__':

    cv2.namedWindow("previewName")
    cam = cv2.VideoCapture(0)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False
    try:
        while rval:
            resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow("previewName", resized)
            rval, frame = cam.read()
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        cv2.destroyWindow("previewName")