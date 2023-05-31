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

def get_keypoints(annotation):
    keypoints = []

    for i in range(len(annotation["shapes"])):
        if annotation["shapes"][i]["shape_type"] == "point":
            point = np.array(annotation["shapes"][i]["points"])
            point = np.append(point, 1)
            point = np.squeeze(point.tolist())
            keypoints.append(point)

    keypoints = np.array(keypoints)

    for i in range(len(keypoints)):
        keypoints[i][0] = int(keypoints[i][0])
        keypoints[i][1] = int(keypoints[i][1])
        keypoints[i][2] = int(keypoints[i][2])

    keypoints = np.array(keypoints)
    keypoints = np.expand_dims(keypoints, axis = 0)
    return keypoints

def create_annotations(images, path):
# Looping through all training images to create theirs individual annotations
    for img in images:
        img_name = img[:img.rfind(".")]

        for annName in old_anns:
            ann = load_annotation(join(OLD_ANNOTATIONS, annName))

            ann_short_name = annName[:annName.rfind(".")]

            if ann_short_name == img_name:
                keypoints = get_keypoints(ann)

                annotation = {
                    "keypoints" : keypoints.tolist()
                }
                annotation = json.dumps(annotation)

                save = open(join(path, "annotations", annName), "w")
                save.write(annotation)
                save.close()

create_annotations(train_imgs, TRAIN_PATH)
create_annotations(test_imgs, TEST_PATH)