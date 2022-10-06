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

classification_model = TabletsModel(6)
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


def anomaly_detection(image,blister_dict,blister_coords):

    for key,value in blister_coords.items():

        blister_img=image[value[1]:value[3],value[0]:value[2]]
        
        # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'.png',cropped_img)

        segmented_tablet,mask= segmentator.segment(model=tablets_segmentation, img=blister_img, threshold_value=float(0.5))

        cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'segmentation.png',segmented_tablet)

        
        if len(np.where(mask==255)[0])>int(os.getenv('segmentation_mask_threshold')):



            # print(f'blister keys----{blister_dict[key]}')

            for key1,value1 in blister_dict[key].items():

                # print(key1)
                # print(value1)

                cropped_pocket=segmented_tablet[value1[1]:value1[3],value1[0]:value1[2]]
                cropped_mask=mask[value1[1]:value1[3],value1[0]:value1[2]]
                # print(len(np.where(cropped_mask==255)[0]))


                # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'crop_pocket.png',cropped_pocket)

                rotated_tablet=rotate_tablets(cropped_pocket,blister_dict,key,key1)

                # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'rotated_tablet.png',rotated_tablet)

                segmented_tablet[value1[1]:value1[3],value1[0]:value1[2]]=rotated_tablet

            # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'segmentation.png',segmented_tablet)


            prediction = utils3.make_prediction(model=tablets_anomaly_model, img=segmented_tablet)
                    # Postprocessing the prediction of model
            defect_mask, heat_map = utils3.postprocess(anomaly_map=prediction["anomaly_map"], img=segmented_tablet, threshold=float(os.getenv('anomaly_mask_threshold')))
                    
            # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'defect_mask.png',defect_mask)
            cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'haet_map.png',heat_map)
            # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'defectmask.png',defect_mask)
            # return image,blister_img,defect_mask,blister_dict,blister_coords


                    # # Saving final result
                    # x, y = np.where(defect_mask==255)
                    # # Checking amount of defected area
                    # if len(x) > int(200):

                    #     print('defected')
                    #     print('passing it into classification')
                    #     cropped_pil_img= cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    #     cropped_pil_img=Image.fromarray(cropped_pil_img)
                    #     # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'.png',cropped_img)

                    #     defect=predict_image(cropped_pil_img,classification_model)


                    #     if defect=='good':
                    #         count+=1
                    
                    # # Visualizing defect on matrix

                    #     final_res = utils3.visualize_defect(segmented_tablet, defect_mask)
                        # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'final_res.png',final_res)
                # else:
                #     print('pocket is empty')
        else:
            print('blister is empty')
            cv2.rectangle(image,(value[0],value[1]),(value[2],value[3]),(0,0,255),2)
            cv2.putText(image,str(key), (value[0], value[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            defect_mask=segmented_tablet.copy()

            # return image,blister_img,defect_mask,blister_dict,blister_coords
    return image,blister_img,defect_mask,blister_dict,blister_coords
   


############################classification###################################################################

def tablet_classification(image,blister_img,defect_mask,blister_dict,blister_coords):

    for key,value in blister_coords.items():

        for key1,value1 in blister_dict[key].items():

            if len(value1)>4:
            
                masked_img=defect_mask[value1[1]:value1[3],value1[0]:value1[2]]
                
                tablet_mask=masked_img[value1[5]:value1[7],value1[4]:value1[6]]
                
                cropped_img=blister_img[value1[1]:value1[3],value1[0]:value1[2]]

                cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+str(key1)+'.png',cropped_img)

                print(len(np.where(tablet_mask==255)[0]),'mask_length')

                if len(np.where(tablet_mask==255)[0])>int((os.getenv('tablet_mask_threshold'))):

                    cropped_pil_img= cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    cropped_pil_img=Image.fromarray(cropped_pil_img)

                    defect=predict_image(cropped_pil_img,classification_model)

                    print(f'name of the defect---{defect}')

                    if defect=='good':
                        cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,255,0),2)
                        cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)


                    else:
                        cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,0,255),2)
                        cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                else:
                    print('its good')
                    cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,255,0),2)
                    cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    print(f'its a good_tablet----{key}{key1}') 
                    
            else:
                cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,0,255),2)
                cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                print('Its an empty_pocket')
                
    return image

####################################################main#########################################################

def main(img_path):
    
    image = cv2.imread(img_path)
    
    x=time.time()

    blister_dict,blister_coords=pocket_detection(image)

    image,blister_img,defect_mask,blister_dict,blister_coords=anomaly_detection(image,blister_dict,blister_coords)

    image=tablet_classification(image,blister_img,defect_mask,blister_dict,blister_coords)
    
    cv2.imwrite('/home/frinks3/RISHABH/ACG/'+'result.png',image)

    print(f'third----{time.time()-x}')
    
    # print(f'time----{end-start} seconds')

############################################################################################################


main('/home/frinks3/RISHABH/ACG/new/testing/Chipping (6).bmp')







    # img_path='/home/frinks3/RISHABH/ACG/new/Good.png'

