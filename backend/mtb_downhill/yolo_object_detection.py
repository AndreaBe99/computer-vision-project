import cv2
import numpy as np
from config import *

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1]
                            for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1]
                            for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, classes, class_id, 
                    confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    color = COLORS[class_id]
    img_copy = img.copy()
    cv2.rectangle(img_copy, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img_copy, label, (x-10, y-10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('YOLO Bike Detection', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def yolo_object_detection(image):
    image_height, image_width, _ = image.shape
    scale = 0.00392

    net = cv2.dnn.readNet(YoloDetectorEnum.WEIGHTS.value,
                          YoloDetectorEnum.CONFIG.value)

    blob = cv2.dnn.blobFromImage(image, 
                                    scale, 
                                    (416, 416), 
                                    (0, 0, 0), 
                                    True, 
                                    crop=False)

    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids, confidences, boxes = [], [], []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, 
                                confidences, 
                                conf_threshold, 
                                nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        x, y, w, h = box
        if str(YoloDetectorEnum.CLASSES.value[class_id]) == "bike":
            break
    
    if DEBUG:
        draw_prediction(image, YoloDetectorEnum.CLASSES.value, class_ids[i], confidences[i],
                    round(x), round(y), round(x+w), round(y+h))

    # Crop the image, but firt aument the size of the box
    x = round(x) - 20
    y = round(y)
    x_plus_w = round(x+w) + 30
    y_plus_h = round(y+h) + 30

    cropped_image = image[y:y_plus_h, x:x_plus_w]

    return [(x, y), (x_plus_w, y_plus_h)]
