
import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

# this python file is built off
# https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400
# https://github.com/matterport/Mask_RCNN

# to install on windows
# install python 3.7.5 64bit (do not install 3.8)
# pip3 install --upgrade pip
# pip3 install numpy
# pip3 install opencv-python
# pip3 install mrcnn
# pip3 install scipy
# pip3 install scikit-image
# pip3 install keras
# pip3 install tensorflow==1.15 (mrcnn/keras requires older version)
# note tensorflow 1.15 installs both GPU/CPU versions

# to run
# py parking.py


# anther interesting resource is here
# http://cnrpark.it/
# https://github.com/fabiocarrara/deep-parking

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Video file or camera to process - set this to 0 to use your webcam instead of a video file
VIDEO_SOURCE = "images/parking-small.m4v"
VIDEO_SOURCE = "images/parking.mp4"
VIDEO_SOURCE = "images/Parking-1.m4v"

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
parked_car_boxes = None

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# start off with a higher weight to keep detection sticky
detection_stickiness_weight = 10


def getmiddle (p1, p2):
    p = 0
    if (p1 <= p2):
        p = p1 + abs(p1 - p2)/2
    else:
        p = p2 + abs(p1 - p2)/2
    return int(p)

def cleanup_boxes(tracked_boxes):
    # sometimes a new box will come in and be smaller than the orignal one
    # such that the overlap doesn't get a high number (because we're checking if
    # new boxes overlap old ones). It would give a good overlap number if
    # the lists were switched
    # so now we're stuck with two boxes occupying the same space.
    # we can check overlap against itself and remove boxes
    nptracked = np.array(tracked_boxes) # convert for overlap call
    overlaps = mrcnn.utils.compute_overlaps(nptracked, nptracked)
    a_idx = 0
    boxes_to_remove = []
    clean_tracked_boxes = []
    for ov in overlaps:
        b_idx = 0
        remove = False
        for v in ov:
            # strong overlap and not the exact same box - remove dup
            if v > 0.65 and b_idx != a_idx:
                print ("remove box ", b_idx, " from", len(tracked_boxes))
                boxes_to_remove.append(b_idx)
                remove = True
            b_idx += 1
        if remove == False:
            print("keeping box ", a_idx, " ", tracked_boxes[a_idx], "\nov: ", ov)
            clean_tracked_boxes.append(tracked_boxes[a_idx])
        a_idx += 1
    return clean_tracked_boxes

def track_boxes (cars, remembered, f):
    # we want to find a similar car box in our
    # 'remembered' boxes. If a box has remained in a
    # similar/same position for a certain number of frames
    # we can declare this box a parking spot

    tracked_boxes = []
    potential_newboxes = [True] * len(cars)

    rboxes = []
    if len(remembered[0]) == 5: # need to remove the weights
        for rbox in remembered:
            y1, x1, y2, x2, w = rbox
            rboxes.append(np.array([y1, x1, y2, x2]))
        rboxes = np.array(rboxes) # convert from list to np.array (for compute_overlaps)
    else: # first time without weight
        rboxes = remembered

    # find overlaps
    overlaps = mrcnn.utils.compute_overlaps(rboxes, cars)

    # go through overlaps and find which car significantly overlaps a remembered box
    # overlaps is a 2D array
    # rows are remembered, columns is how much that space was overlaps by cars

    r_index = 0
    for ov in overlaps:
        car_index = 0
        found_similar = False
        #print (ov):
        for v in ov:
            #print (len(ov), " ", len(cars), " ", car_index, " ", len(overlaps), " ", len(remembered), " ", r_index)
            if v > 0.85:
                # cars[car_index] significantly overlaps remembered[r_index]
                # save the average box
                #print ("car ", car_index, " ", cars[car_index], " matches ", r_index, " ", remembered[r_index])
                carbox = cars[car_index]
                rbox = remembered[r_index]
                y1 = getmiddle(carbox[0], rbox[0])
                x1 = getmiddle(carbox[1], rbox[1])
                y2 = getmiddle(carbox[2], rbox[2])
                x2 = getmiddle(carbox[3], rbox[3])
                if len(rbox) == 5:
                    weight = rbox[4]
                else:
                    weight = detection_stickiness_weight - 1 # start off higher
                weight += 1
                avgbox = np.array([y1, x1, y2, x2, weight])
                tracked_boxes.append(avgbox)
                found_similar = True
                print ("found similar ", r_index, " ", avgbox)
                potential_newboxes[car_index] = False
                break # no point continuing
            car_index += 1

        if found_similar == False:
            # didn't find any similar - keep the old one
            # but decr weight
            rbox = remembered[r_index]
            w = 0
            if len(rbox) == 5:
                y1, x1, y2, x2, w = rbox
                w -= 1
            # if it's missing from the detected cars it can eventually go negative
            # moving cars can make this sort of thing happen
            if w >= 0:
                oldbox = np.array([y1, x1, y2, x2, w])
                tracked_boxes.append(oldbox)
                print("kept old box ", w, " ", oldbox)
        r_index += 1

    # go through all the detected boxes and see if there were
    # any new ones
    car_index = 0
    for pnew in potential_newboxes:
        if pnew == True:
            t = cars[car_index]
            y1, x1, y2, x2 = t
            weight = detection_stickiness_weight # start off higher so it doesn't immediately dissappear
            newbox = np.array([y1, x1, y2, x2, weight])
            print ("adding new ", newbox)
            tracked_boxes.append(newbox)
        car_index += 1

    #tracked_boxes = cleanup_boxes(tracked_boxes)

    print ("tracked ", len(tracked_boxes))
    return tracked_boxes

def update_parking_spots (remembered, spots):
    # based on our remembered boxes, lets
    # accumulate the weights in our parking spots
    return



frame_cnt = 0
process_frame_limit = 1  # skip frames for processing

class ParkingSpot:
    def __init__(self, box=[], weight=0, status=0):
        self.box = box
        self.weight = weight
        self.status = status

parking_spots = []
remembered_carboxes = []

# Loop over each frame of video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # process every process_frame_limit
    frame_cnt += 1
    if frame_cnt % process_frame_limit != 0:
        continue

    print("processing frame ", frame_cnt)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]

    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)

    # Filter the results to only grab the car / truck bounding boxes
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])

    if len(remembered_carboxes) == 0:
        remembered_carboxes = car_boxes
    remembered_carboxes = track_boxes(car_boxes, remembered_carboxes, frame_cnt)

    print ("\n\n\n")
    print("Cars found in frame of video: ", len(car_boxes))
    print("tracked boxes ", len(remembered_carboxes))

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 1

    # Draw each box on the frame
    b_idx = 0
    for box in remembered_carboxes:

        y1, x1, y2, x2, w = box

        #print("Car: ", box, "w ", w)

        # Draw the box
        red = (0, 0, 255)
        green = (0, 255, 0)
        colour = green
        if w != detection_stickiness_weight:
            colour = red

        # only display parking spots
        # weight > 15
        #if w >= 15:
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)

        cv2.putText(frame, str(b_idx) + ":" + str(w),
            (x1, y1),
            font,
            fontScale,
            fontColor,
            lineType)
        b_idx += 1


    # Show the frame of video on the screen
    cv2.imshow('Video', frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
cv2.destroyAllWindows()
