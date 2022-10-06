#from tkinter import E
import cv2
import numpy as np
import os
import time
import math

from anomalib.post_processing import post_process, superimpose_anomaly_map
# from segmentation.segment import Segmentator


# Function to get prediction from anomalib_model
def make_prediction(model, img):
    # Preprocessing the image
    processed_img = model.pre_process(img)
    # Prediction
    res = model.forward(processed_img)
    # Post processing the output
    prediction  = model.post_process(res)
    return prediction


# Function to do gamma correction
def do_gamma_correction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)
    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)
    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
    return img_gamma


# Function to crop matrix
def crop_matrix(img):
  contours, _ = cv2.findContours(img[:,:, 0], 
      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  x,y,w,h = cv2.boundingRect(contours[0])
  cropped_matrix = img[y:y+h, x:x+w]
  return cropped_matrix


# Function that post process the output of anomalib model
def postprocess(anomaly_map, img, threshold):
    actual_img = cv2.resize(img,(256,256))
    # Generating mask of defect
    defect_mask = post_process.compute_mask(anomaly_map, threshold)
    # Generating the heatmap
    heat_map = superimpose_anomaly_map(anomaly_map, actual_img)
    # Converting heatmap
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_RGB2BGR)
    # contours, hierarchy = cv2.findContours(defect_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # final_res = cv2.drawContours(actual_img, contours, -1, (0,255,0), 1)
    return defect_mask, heat_map

# FUnction visualise the damage on matrix
def visualize_defect(img, defect_mask, res_dict):
    actual_img = cv2.resize(img,(256,256))
    if res_dict["matrix_defected"]:
        contours, hierarchy = cv2.findContours(defect_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_res = cv2.drawContours(actual_img, contours, -1, (0,255,0), 1)
        return final_res
    else:
        return actual_img

