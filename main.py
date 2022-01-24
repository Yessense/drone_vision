import numpy as np
import cv2


capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

threshold = 0.6
nms_threshold = 0.2
color = (255, 0, 0)
class_names = []
class_file = 'coco.names'

with open(class_file, 'rt') as f:
    class_names = f.read().split('\n')

config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



if __name__ == '__main__':
    while True:
        success, img = capture.read()
        class_ids, confidences, bbox = net.detect(img, confThreshold=threshold, nmsThreshold=nms_threshold)

        if len(class_ids):
            for class_id, conf, box in zip(class_ids.flatten(), confidences.flatten(), bbox):
                x, y, w, h = box
                color_image = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, class_names[class_id - 1], (x + 10, y), 0, 1.0, (0, 255, 0))

        # Show images
        cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Camera', img)
        cv2.waitKey(1)
