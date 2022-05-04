import argparse
import os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.general import download_and_unzip, run_inference
from skimage.io import imread_collection
from collections import Counter
import time
from pathlib import Path    
from pymediainfo import MediaInfo

default_model = './data/yolov5s_saved_model/'
window_name = 'Yolov5 object detection'
detection_folder = './detections/'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)       

def run_model(model, image_BGR, prob, thres):
    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    blob = np.transpose(blob, (0, 2, 3, 1))
    pred = model(blob)
    h, w = (image_BGR.shape[:2])
    return run_inference(pred, prob, thres, h, w)

def print_results(results, class_numbers, labels, calctime):
    counter = 1
    if len(results) > 0:
        print(f"\nDETECTION DONE (in %.2f ms): \n----------------\n" % calctime)
        for i in results.flatten():
            print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
            counter += 1
    else:
        print('No objects detected.')

def draw_bounding_boxes(frame, results, bounding_boxes, confidences, class_numbers, labels, colours):
    if len(results) > 0:
        for i in results.flatten():
            colour_box = colours[class_numbers[i]].tolist()
            # draw rectangle
            xmin, ymin = bounding_boxes[i][0], bounding_boxes[i][1]
            xmax, ymax= bounding_boxes[i][2], bounding_boxes[i][3]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), colour_box, 2)
            # draw text
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],confidences[i])
            cv2.putText(frame, text_box_current, (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box, 2)

def run_image(image, to_save_path, model, labels, prob, thres, visualize, colours):
    t1 = time.time()
    results, bounding_boxes, confidences, class_numbers = run_model(model, image, prob, thres)
    t2 = time.time() - t1
    print_results(results, class_numbers, labels, t2)
    if(visualize):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        draw_bounding_boxes(image_rgb, results, bounding_boxes, confidences, class_numbers, labels, colours)
        h, w = (image_rgb.shape[:2])
        cv2.resizeWindow(window_name, h, w) 
        cv2.imshow(window_name, image_rgb)
        cv2.imwrite(to_save_path, image_rgb)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

def run_video(cap, camera, model, labels, prob, thres, visualize, colours):
    # Check if video opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if(camera):
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        if ret == True:
            # Apply the model on the frame
            t1 = time.time()
            results, bounding_boxes, confidences, class_numbers = run_model(model, frame, prob, thres)
            t2 = time.time() - t1
            print_results(results, class_numbers, labels, t2)
            if(visualize):
                draw_bounding_boxes(frame, results, bounding_boxes, confidences, class_numbers, labels, colours)
            # Display the resulting frame
            t3 = time.time() - t1
            fps = "FPS: {fps:.2f}".format(fps = 1.0 / t3)
            cv2.putText(frame, fps, (10,25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            h, w = (frame.shape[:2])
            cv2.imshow(window_name, frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def main(
    saved_model=None,
    labels='./data/coco.names',
    source='./data/images/',
    prob=0.7,
    thres=0.3,
    visualize=True
):
    # Load model:
    if(saved_model is None):
        if(os.path.isdir(default_model)):
            print('No saved model is given. Using default Yolov5s coco dataset.')
        else:
            print('No saved model is given. Downloading the Yolov5s coco dataset.')
            download_and_unzip('https://github.com/Jeremy-vdw/Yolov5-tf2/releases/download/models/yolov5s_saved_model.zip', './data/')
        saved_model = default_model

    model = tf.saved_model.load(saved_model)

    # Loading labels
    with open(labels) as f:
        labels = [line.strip() for line in f]
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    #make detections folder
    if not os.path.isdir(detection_folder):
        os.mkdir(detection_folder)
        

    if(os.path.isdir(source)):
        ##folder is used.
        pngs = source + '/*.png'
        jpgs = source + '/*.jpg'
        image_list = imread_collection([pngs, jpgs])
        for i, image in enumerate(image_list):
            run_image(image, (detection_folder + Path(image_list.files[i]).name), model, labels, 
                prob, thres, visualize, colours)

    elif(os.path.isfile(source)):
        ## file is used.
        fileInfo = MediaInfo.parse(source)
        for track in fileInfo.tracks:
            if track.track_type == "Image":
                image = cv2.imread(source)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                run_image(image, (detection_folder + Path(source).name), model, labels, 
                prob, thres, visualize, colours)

            elif track.track_type == "Video":
                cap = cv2.VideoCapture(source)
                run_video(cap, False, model, labels, prob, thres, visualize, colours)
    elif(source == '0'):
        ## camera is used.
        cap = cv2.VideoCapture(0)
        run_video(cap, True, model, labels, prob, thres, visualize, colours)


            



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str, help='location where saved model is stored.')
    parser.add_argument('--labels', type=str, default='./data/coco.names', help='location where dataset.names is stored.')
    parser.add_argument('--source', type=str, default='0', help='location where images or video is stored. Use 0 for webcam.')
    parser.add_argument('--prob', type=float, default=0.7, help='minimum probability to eliminate weak predictions.')
    parser.add_argument('--thres', type=float, default=0.3, help='setting threshold for filtering weak bounding boxes with NMS.')
    parser.add_argument('--visualize', type=bool, default=True, help='draw bounding boxes or not.')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))