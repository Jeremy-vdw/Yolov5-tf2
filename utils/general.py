from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import cv2

def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)


def YOLOdetect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = np.array(output_data[0])  # x(1, 25200, 85) to x(25200, 85)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4] xywh
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    #xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]
    return boxes, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

def run_inference(prediction, prob, thres, h, w):
    boxes, classes, scores = YOLOdetect(prediction)
    boxes = xywh2xyxy(boxes)

    bounding_boxes = []
    confidences = []
    class_numbers = []

    #get relevant boxes
    for i in range(0,25200):
        score = scores[i]
        class_current = classes[i]
        if score > prob:
            
            box_current = boxes[i] * np.array([w, h, w, h])

            xmin = int(box_current[0])
            ymin = int(box_current[1])
            xmax = int(box_current[2])
            ymax = int(box_current[3])

            bounding_boxes.append([xmin, ymin, xmax, ymax])
            confidences.append(float(score))
            class_numbers.append(class_current)
    
    confidences = list(map(float, confidences)) ##confidences to float instead of npfloat

    #apply NMS
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, prob, thres)

    return results, bounding_boxes, confidences, class_numbers

