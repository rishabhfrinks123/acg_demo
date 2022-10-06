import numpy as np
import os
import cv2
# from nympy import source
import torch
import time

# before = time.time()
# label_names = ['pocket']

# model_path = "/home/frinks3/RISHABH/ACG/yolo_pockets.pt"
# model = torch.hub.load('/home/frinks3/RISHABH/ACG/new/yolov5', "custom", source='local',path =  model_path, force_reload=True)

def create_blister_dict(bboxes):
    # Sorting the centers based on y_axis
    new_bboxes = sorted(bboxes, key = lambda x: x[1])
    diffs = []
    for i in range(len(new_bboxes)-1):
        diffs.append(abs(new_bboxes[i][1] - new_bboxes[i+1][1]))
    # Sorting pockets from left to right
    pockets = []
    start = 0 
    for index in [i for i, val in enumerate(diffs) if val > 100]:
        sorted_row = sorted(new_bboxes[start:index+1], key = lambda x: x[0])
        pockets.extend(sorted_row)
        start = index+1
    pockets.extend(sorted(new_bboxes[start:] , key = lambda x: x[0]))
    # Creating pockets dictionary
    pockets_dict = {}
    for index, pocket in enumerate(pockets):
        pockets_dict[index+1] = pocket[2:]
    # Creating blister dictionary
    blister_dict = {"blister1": {**dict(list(pockets_dict.items())[:5]), **dict(list(pockets_dict.items())[10:15])},
    "blister2": {**dict(list(pockets_dict.items())[5:10]), **dict(list(pockets_dict.items())[15:20])},
    "blister3": {**dict(list(pockets_dict.items())[20:25]), **dict(list(pockets_dict.items())[30:35])},
    "blister4": {**dict(list(pockets_dict.items())[25:30]), **dict(list(pockets_dict.items())[35:])}}

    # Finding blister cordinates
    blister_cord = {}
    for key in blister_dict.keys():
        blister_cord[key] = [val-20 for val in list(blister_dict[key].items())[0][1][:2]]+[val+20 for val in list(blister_dict[key].items())[-1][1][2:]]
    # print(blister_cord)
    # Finding cordinates of bbox wrt blisters
    for key, value in blister_cord.items():
        norm_cord = value[:2]
        for pocket_key in blister_dict[key].keys():
            temp_cords = blister_dict[key][pocket_key]
       
            blister_dict[key][pocket_key] = [temp_cords[0]-norm_cord[0], temp_cords[1]-norm_cord[1], temp_cords[2]-norm_cord[0], temp_cords[3]-norm_cord[1]]
    # print(blister_dict)

    return blister_dict, blister_cord

def find_bboxes(img,model):
    result = model(img)
    labels, cord = result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]
    x_shape, y_shape = img.shape[1], img.shape[0]
    bboxes = []
    for i in range(len(labels)):
        row = cord[i]
        if row[4] >= 0.7:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            bboxes.append([int((x1+x2)/2), int((y1+y2)/2), x1, y1, x2, y2])
    return bboxes

# if __name__ == "__main__":
#     img_path = "/home/amal/Frinks/ACG/yolo_Data/circle_detection/images/Good (81)_50augumented.png"

#     img = cv2.imread(img_path)
#     bboxes = find_bboxes(img)
#     before = time.time()
#     final_dict =  create_blister_dict(bboxes)
#     print(time.time() - before)
#     print("Done")
