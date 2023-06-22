import json

def is_box(val):
    return val['shape_type'] == 'rectangle'

def is_box_point(val, groupid):
    return val['shape_type'] == 'point' and val['group_id'] == groupid

def labelme_to_coco(labelme_data):
    shapes = labelme_data["shapes"]
    image_width = labelme_data["imageWidth"]
    image_height = labelme_data["imageHeight"]
    bboxes = [shape for shape in shapes if is_box(shape)]

    out_boxes = []
    out_points = []
    for bbox in bboxes:
        group_id = bbox['group_id']

        x1, y1 = bbox['points'][0]
        x2, y2 = bbox['points'][1]

        if x1 > x2:
            tmp = x2
            x2 = x1
            x1 = tmp

        if y1 > y2:
            tmp = y2
            y2 = y1
            y1 = tmp
        
        int_box = [int(item) for item in [x1, y1, x2, y2]] # convert to integers
        out_boxes.append(int_box)

        box_keypoints = [shape for shape in shapes if is_box_point(shape, group_id)]
        box_keypoints = sorted(box_keypoints, key = lambda x: x['label'] )

        my_points = []    
        for point in box_keypoints:
            my_point = [int(item) for item in point['points'][0]]
            my_point.append(1)
            my_points.append(my_point)    
        
        out_points.append(my_points)

    coco_data = {
        "bboxes": out_boxes,
        "keypoints": out_points
    }
    
    return coco_data


def convert_labelme_to_coco(input_path, output_path):
    with open(input_path, "r") as file:
        labelme_data = json.load(file)
        
    coco_data = labelme_to_coco(labelme_data)
    
    with open(output_path, "w") as file:
        json.dump(coco_data, file, indent=4)

# Example usage:
LABELME_FOLDER = r"./rv12_dataset_v2" #With annotated data 
OUTPUT_FOLDER = r"./rv12_COCO_dataset"

import os
jsons = [x for x in os.listdir(LABELME_FOLDER) if ".json" in x ]
for json_name in jsons:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    input_path = os.path.join(LABELME_FOLDER, json_name)
    output_path = os.path.join(OUTPUT_FOLDER, json_name)
    convert_labelme_to_coco(input_path, output_path)
print("Conversion complete.")