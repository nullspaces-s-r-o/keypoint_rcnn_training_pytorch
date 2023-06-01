import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import json

TRAIN_PATH = "train"
TEST_PATH = "test"

OLD_ANNOTATIONS = "annotations"

# Getting old annotations
old_anns = [ann for ann in listdir(OLD_ANNOTATIONS) if isfile(join(OLD_ANNOTATIONS, ann))]

# Getting all image names
train_imgs = [img for img in listdir(join(TRAIN_PATH, "images")) if isfile(join(TRAIN_PATH, "images", img))]
test_imgs = [img for img in listdir(join(TEST_PATH, "images")) if isfile(join(TEST_PATH, "images", img))]

# Loading annotation
def load_annotation(path):
    f = open(path)
    ann = json.load(f)
    return ann

def get_bbox(annotation):
    for ann in annotation["shapes"]:
        if ann["shape_type"] == "rectangle":
            bbox = np.array(ann["points"]).flatten()

            # Check if points are in order x_min, y_min, x_max, y_max
            if bbox[0] > bbox[2]:
                temp = bbox[0]
                bbox[0] = bbox[2]
                bbox[2] = temp
            if bbox[1] > bbox[3]:
                temp = bbox[1]
                bbox[3] = bbox[1]
                bbox[1] = temp

            bbox = [int(axis) for axis in bbox]

    bbox = np.array(bbox)
    bbox = np.expand_dims(bbox, axis = 0)
    return bbox

def get_keypoints(annotation):
    keypoints = []

    for ann in annotation["shapes"]:
        if ann["shape_type"] == "point":
            point = np.array(ann["points"])
            point = np.append(point, 1)
            point = np.squeeze(point.tolist())
            keypoints.append(point)

    keypoints = np.array(keypoints)

    for i in range(len(keypoints)):
        keypoints[i][0] = int(keypoints[i][0])
        keypoints[i][1] = int(keypoints[i][1])
        keypoints[i][2] = int(keypoints[i][2])

    keypoints = np.expand_dims(keypoints, axis = 0)
    return keypoints

def create_annotations(images, path):
    missings = []

    # Looping through all training images to create theirs individual annotations
    for img in images:
        img_name = img[:img.rfind(".")]

        for annName in old_anns:
            ann = load_annotation(join(OLD_ANNOTATIONS, annName))

            ann_short_name = annName[:annName.rfind(".")]

            if ann_short_name == img_name:
                keypoints = get_keypoints(ann)
                bbox = get_bbox(ann)

                annotation = {
                    "bboxes" : bbox.tolist(),
                    "keypoints" : keypoints.tolist()
                }
                annotation = json.dumps(annotation)

                save = open(join(path, "annotations", annName), "w")
                save.write(annotation)
                save.close()
                break
        else:
            missings.append(ann_short_name)

    new_list = []
    dup_list = []
    for i in missings:
        if i not in new_list:
            new_list.append(i)
        else:
            dup_list.append(i)
    if len(dup_list) != 0:
        print(f"{dup_list} are not annotated")

create_annotations(train_imgs, TRAIN_PATH)
create_annotations(test_imgs, TEST_PATH)