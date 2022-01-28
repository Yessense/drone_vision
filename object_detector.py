from typing import List

import numpy as np
import cv2

threshold = 0.6
nms_threshold = 0.2

class_file = 'coco.names'
with open(class_file, 'rt') as f:
    class_names: List[str] = f.read().split('\n')

color = (255, 0, 0)


config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def get_objects(img, draw=True, objects: List = None):
    if objects is None:
        objects = class_names
    class_ids, confidences, bbox = net.detect(img, confThreshold=threshold, nmsThreshold=nms_threshold)

    object_info = []

    if len(class_ids):
        for class_id, conf, box in zip(class_ids.flatten(), confidences.flatten(), bbox):
            if draw:
                class_name = class_names[class_id - 1]
                object_info.append([box, class_name])
                x, y, w, h = box
                img = cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=2)
                cv2.putText(img, class_name, (x + 10, y), 0, 1.0, (0, 255, 0))
    print(object_info)
    return img, object_info
