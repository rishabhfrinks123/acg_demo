# pip install mtm
import cv2
import time
import numpy as np
import math
from scipy import ndimage
from scipy.spatial import distance as dist
import os
import torch
import mtm
print("mtm version : ", mtm.__version__)

from mtm import matchTemplates
from mtm.detection import plotDetections

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from defects_classification import TabletsModel,to_device,get_default_device,predict_image
from PIL import Image
from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.torch_inference import TorchInferencer
from segmentation.segment import Segmentator
from segmentation.model import PetModel
import station3_utils as utils3
from dotenv import load_dotenv
# import socketio
import warnings
warnings.filterwarnings("ignore")
from rotation import rotate_tablets


# img=cv2.imread('/home/rishabh/frinks/ACG_tablets/ACGI_Grey On Grey Images_T&V Sets_7Sept2022/Good (177).bmp')

# # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# temp0=img[607:790,421:540]

# cv2.imwrite('temp.bmp',temp0)

device = get_default_device()


tablets_segmentation = PetModel("UnetPlusPlus", "efficientnet-b5", in_channels=3, out_classes=1)
# Loading the saved model weights
tablets_segmentation.load_state_dict(torch.load("/home/frinks3/RISHABH/ACG/tablet_wogamma.pth"))
tablets_segmentation.state_dict()
to_device(tablets_segmentation, device)
# tablets_segmentation.eval()

# Loading matrix anomalib model
model_configs = get_configurable_parameters('/patchcore/')
tablets_anomaly_model = TorchInferencer(config=model_configs, model_source='/home/frinks3/RISHABH/ACG/model.ckpt')

# Initialising Segmentator
segmentator = Segmentator()

classification_model = TabletsModel(7)
classification_model.load_state_dict(torch.load('/home/frinks3/RISHABH/ACG/resnet34classes7.pth'))
to_device(classification_model, device)




img_path='/home/frinks3/RISHABH/ACG/new/share_img/Good (237).bmp'
# save_dir='/home/rishabh/frinks/ACG_tablets/ACGI_Grey On Grey Images_T&V Sets_7Sept2022/anomaly_more_crop_data'


# print(os.listdir('/home/rishabh/frinks/ACG_tablets/ACGI_Grey On Grey Images_T&V Sets_7Sept2022/anomaly_more _data'))



temp_img=cv2.imread('/home/frinks3/RISHABH/ACG/new/temp.bmp')

temp0=cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)

listTemplates = [temp0]
listLabels    = ["Temp0"]

for i,angle in enumerate([15,30,45,60]):
    rotated = np.rot90(temp0, k=i+1) 
    # rotated = rotate(temp0, angle)
    listTemplates.append(rotated)
    listLabels.append(str(angle))


image=cv2.imread(img_path)

# image=Image.open(img_path)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

listDetections = matchTemplates(gray, 
                        listTemplates,
                        listLabels,
                        nObjects=40, 
                        scoreThreshold=0.4, 
                        maxOverlap=0.3)


# print(f'image---{img}')
b_box={}

for i in range(len(listDetections)):
    b_box[i]=listDetections[i].get_xywh()

# print(b_box)

bbcords={}

for key,value in b_box.items():
    bbcords[key]=[value[0],value[1],value[2]+value[0],value[3]+value[1]]

print(bbcords)

# image=Image.open(img_path)



count=0
for key,value in bbcords.items():

    cropped_img=image[value[1]:value[3],value[0]:value[2]]
    
    # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'.png',cropped_img)

    segmented_tablet,mask= segmentator.segment(model=tablets_segmentation, img=cropped_img, threshold_value=float(0.5))

    if len(np.where(mask==255)[0])>30:
        
        start=time.time()

        segmented_tablet=rotate_tablets(segmented_tablet)


        end=time.time()

        print(f'time----{end-start}sec')   

        cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'.png',segmented_tablet)

        # Cropping the segmented image to fit the matrix
        # matrix = utils3.crop_matrix(matrix_img)
        # Applying gamma correction to normalise brightness
        # matrix = utils3.do_gamma_correction(img=matrix)
        # Passing the matrix image to the anomalib model

        prediction = utils3.make_prediction(model=tablets_anomaly_model, img=segmented_tablet)
        

        # Postprocessing the prediction of model
        defect_mask, heat_map = utils3.postprocess(anomaly_map=prediction[0], img=segmented_tablet, threshold=float(0.4))
        
        cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'heatmap.png',heat_map)

        # Saving final result
        x, y = np.where(defect_mask==255)
        # Checking amount of defected area
        if len(x) > int(200):

            print('defected')
            print('passing it into classification')
            cropped_pil_img= cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

            cropped_pil_img=Image.fromarray(cropped_pil_img)
            # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'.png',cropped_img)

            defect=predict_image(cropped_pil_img,classification_model)


            if defect=='good':
                count+=1
        
        # Visualizing defect on matrix

            final_res = utils3.visualize_defect(segmented_tablet, defect_mask)
            cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'final_res.png',final_res)
    else:
        print('defect==empty')

   

print(count)


    # cv2.imwrite(save_dir+'/'+dir+'/'+str(i)+img,img_save)






