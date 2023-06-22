import json

def labelme_to_coco(labelme_data):
    shapes = labelme_data["shapes"]
    image_width = labelme_data["imageWidth"]
    image_height = labelme_data["imageHeight"]
    bboxes = []
    keypoints = []

    for shape in shapes:
        label = shape["label"]
        points = shape["points"]
        
        if shape["shape_type"] == "rectangle":
            x1, y1 = points[0]
            x2, y2 = points[1]
            bbox = [x1, y1, x2, y2]
            bboxes.append(bbox)
        elif shape["shape_type"] == "point":
            x, y = points[0]
            if len(keypoints) > 0 and len(keypoints[-1]) == 1:
                keypoints[-1].append([x, y, 1])
            else:
                keypoints.append([[x, y, 1]])

    coco_data = {
        "bboxes": bboxes,
        "keypoints": keypoints
    }
    
    return coco_data


def convert_labelme_to_coco(input_path, output_path):
    with open(input_path, "r") as file:
        labelme_data = json.load(file)
        
    coco_data = labelme_to_coco(labelme_data)
    
    with open(output_path, "w") as file:
        json.dump(coco_data, file, indent=4)


# Example usage:
LABELME_FOLDER = r"./rv12_example_dataset" #With annotated data 
OUTPUT_FOLDER = r"./rv12_COCO_dataset"

import os
jsons = [x for x in os.listdir(LABELME_FOLDER) if ".json" in x ]
for json_name in jsons:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    input_path = os.path.join(LABELME_FOLDER, json_name)
    output_path = os.path.join(OUTPUT_FOLDER, json_name)
    convert_labelme_to_coco(input_path, output_path)
print("Conversion complete.")