from object_detector import *


capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

while True:
    success, img = capture.read()

    img, object_info = get_objects(img)

    # Show images
    # cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('Camera', img)
    # cv2.waitKey(1)
