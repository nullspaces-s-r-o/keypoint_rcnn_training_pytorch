import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import json

TRAIN_PATH = "train"
TEST_PATH = "test"

OLD_ANNOTATIONS = "keypoints.json"

# Creating annotation folders if not existing
try:
    os.mkdir(join(TRAIN_PATH, "annotations"))
except OSError as error:
    print(error)

try:
    os.mkdir(join(TEST_PATH, "annotations"))
except OSError as error:
    print(error)

# Loading old annotations
f = open(OLD_ANNOTATIONS)
old_annotations = json.load(f)

# Getting all image names
train_imgs = [img for img in listdir(join(TRAIN_PATH, "images")) if isfile(join(TRAIN_PATH, "images", img))]
test_imgs = [img for img in listdir(join(TEST_PATH, "images")) if isfile(join(TEST_PATH, "images", img))]

# Scailing keypoints from 0-1 on each axis to either 0-width or 0-height
def scale_keypoints(path, keypoints):
    img = Image.open(path)
    width, height = img.size
    for i in range(len(keypoints)):
        keypoints[i][0] = int(keypoints[i][0] * width)
        keypoints[i][1] = int(keypoints[i][1] * height)
        keypoints[i][2] = int(keypoints[i][2])

    keypoints = np.array(keypoints)
    keypoints = np.expand_dims(keypoints, axis = 0)
    return keypoints

def create_annotations(images, path):
# Looping through all training images to create theirs individual annotations
    for img in images:
        for ann in old_annotations:
            ann_img = ann["img_path"][ann["img_path"].rfind("\\") + 1:]
            if ann_img == img:

                keypoints = scale_keypoints(join(path, "images", img), ann["joints"])

                annotation = {
                    "keypoints" : keypoints.tolist()
                }
                annotation = json.dumps(annotation)

                name = "{}.json".format(img[:img.rfind(".")])
                save = open(join(path, "annotations", name), "w")
                save.write(annotation)
                save.close()

create_annotations(train_imgs, TRAIN_PATH)
create_annotations(test_imgs, TEST_PATH)