#

##########################segmentation and anomaly#########################'


# def anomaly_detection(image,blister_dict,blister_coords):

#     for key,value in blister_coords.items():

#         blister_img=image[value[1]:value[3],value[0]:value[2]]


###############################################################################

# for img in os.listdir(img_dir):
#     image=cv2.imread(img_dir+'/'+img)
#     img=img.split('.')[0]+'.png'
#     print(img)
#     blister_dict,blister_coords=pocket_detection(image)
    
#     for key,value in blister_coords.items():
#         blister_img=image[value[1]:value[3],value[0]:value[2]]
#         cv2.imwrite('/home/frinks3/RISHABH/ACG/empty_results'+'/'+str(key)+img,blister_img)




# pip install mtm
import cv2
import time
import numpy as np
import math
from scipy import ndimage
from scipy.spatial import distance as dist
import os
import torch

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from defects_classification import TabletsModel,to_device,get_default_device,predict_image
from PIL import Image
from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
from segmentation.segment import Segmentator
from segmentation.model import PetModel
import station3_utils as utils3
from dotenv import load_dotenv
# import socketio
import warnings
warnings.filterwarnings("ignore")
from rotation import rotate_tablets
from sort_bboxes import create_blister_dict,find_bboxes

load_dotenv()
# img=cv2.imread('/home/rishabh/frinks/ACG_tablets/ACGI_Grey On Grey Images_T&V Sets_7Sept2022/Good (177).bmp')

# # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# temp0=img[607:790,421:540]

# cv2.imwrite('temp.bmp',temp0)

device = get_default_device()


tablets_segmentation = PetModel("UnetPlusPlus", "efficientnet-b5", in_channels=3, out_classes=1)
to_device(tablets_segmentation, device)

# Loading the saved model weights
tablets_segmentation.load_state_dict(torch.load(os.getenv('tablet_segmentation_path')))
tablets_segmentation.state_dict()
# to_device(tablets_segmentation, device)
# tablets_segmentation.eval()

# Loading matrix anomalib model
model_configs = get_configurable_parameters('/patchcore/')
tablets_anomaly_model = TorchInferencer(config=model_configs, model_source=os.getenv("anomaly_model_path"))

# Initialising Segmentator
segmentator = Segmentator()

classification_model = TabletsModel(7)
classification_model.load_state_dict(torch.load(os.getenv('classification_model_path')))
to_device(classification_model, device)

yolo_model_path = os.getenv('yolo_model_path')
yolo_model = torch.hub.load(os.getenv('yolo_directory_path'), "custom", source='local',path =yolo_model_path, force_reload=True)





##########################yolo##################################

def pocket_detection(image):
    bboxes = find_bboxes(image,yolo_model)
    blister_dict ,blister_coords =  create_blister_dict(bboxes)
    print("Done")
    return blister_dict,blister_coords


##########################segmentation and anomaly#########################'

# def anomaly_detection(image,name):

def anomaly_detection(image,blister_img,blister_dict,blister_coords,key,value,name,directory):

    # for key,value in blister_coords.items():

    #     blister_img=image[value[1]:value[3],value[0]:value[2]]
        
        # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'.png',cropped_img)

        # segmented_tablet,mask= segmentator.segment(model=tablets_segmentation, img=blister_img, threshold_value=float(0.5))

            segmented_tablet,mask= segmentator.segment(model=tablets_segmentation, img=blister_img, threshold_value=float(0.5))


        # cv2.imwrite(save_dir+'/'+name,segmented_tablet)

        # blister_anomaly=np.zeros((blister_img.shape[0],blister_img.shape[1],3), np.uint8)

        
        # if len(np.where(mask==255)[0])>int(os.getenv('segmentation_mask_threshold')):



            # print(f'blister keys----{blister_dict[key]}')

            for key1,value1 in blister_dict[key].items():

                # print(key1)
                # print(value1)

                cropped_pocket=segmented_tablet[value1[1]:value1[3],value1[0]:value1[2]]

                cv2.imwrite(save_dir+directory+'/'+{key}+{key1}+img,cropped_pocket)

            #     cropped_mask=mask[value1[1]:value1[3],value1[0]:value1[2]]
            #     # print(len(np.where(cropped_mask==255)[0]))


            #     # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'crop_pocket.png',cropped_pocket)

            #     rotated_tablet=rotate_tablets(cropped_pocket,blister_dict,key,key1)

            #     # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'rotated_tablet.png',rotated_tablet)

            #     segmented_tablet[value1[1]:value1[3],value1[0]:value1[2]]=rotated_tablet

            # cv2.imwrite(save_dir+'/'+name+str(key)+str(key1)+'segmentation.png',segmented_tablet)

            # return cropped_pocket
    
                
                

######################################################################################

img_dir='/home/frinks3/RISHABH/ACG/new_classification_train/'
save_dir='/home/frinks3/RISHABH/ACG/new_classification_train_results/'

for directory in os.listdir(img_dir):

    for img in os.listdir(img_dir+directory):

        path=os.path.join(img_dir,directory,img)

        # print(path)

        image=cv2.imread(path)

        img=img.split('.')[0]+'.png'

        print(img)

        blister_dict,blister_coords=pocket_detection(image)

        for key,value in blister_coords.items():
            
            blister_img=image[value[1]:value[3],value[0]:value[2]]
            # cv2.imwrite(save_dir+'/'+str(key)+img,blister_img)
            anomaly_detection(image,blister_img,blister_dict,blister_coords,key,value,img,directory)
            # cv2.imwrite(save_dir+'/'+str(key)+img,segmented_tablet)


    #     image=cv2.imread(img_dir+'/'+img)
    # img=img.split('.')[0]+'.png'
    # print(img)
    # blister_dict,blister_coords=pocket_detection(image)
    
    # for key,value in blister_coords.items():
    #     blister_img=image[value[1]:value[3],value[0]:value[2]]
    #     # cv2.imwrite(save_dir+'/'+str(key)+img,blister_img)
    #     anomaly_detection(image,blister_img,blister_dict,blister_coords,key,value,img)
    #     # cv2.imwrite(save_dir+'/'+str(key)+img,segmented_tablet)

        

# for img in os.listdir(img_dir):
#     image=cv2.imread(img_dir+'/'+img)
#     # img=img.split('.')[0]+'.png'
#     blister_dict,blister_coords=pocket_detection(image)
#     anomaly_detection(image,blister_dict,blister_coords)
