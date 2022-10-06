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
model_configs = get_configurable_parameters('/padim/')

# model_configs = get_configurable_parameters('/patchcore/')
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


def anomaly_detection(image,blister_img,blister_dict,blister_coords,key,value,f,name):

    # for key,value in blister_coords.items():

    #     blister_img=image[value[1]:value[3],value[0]:value[2]]
        
        # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'.png',cropped_img)

        segmented_tablet,mask= segmentator.segment(model=tablets_segmentation, img=blister_img, threshold_value=float(0.5))

        # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'segmentation.png',segmented_tablet)

        # blister_anomaly=np.zeros((blister_img.shape[0],blister_img.shape[1],3), np.uint8)

        
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

                # blister_anomaly=

            # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'segmentation.png',segmented_tablet)


            prediction = utils3.make_prediction(model=tablets_anomaly_model, img=segmented_tablet)
                    # Postprocessing the prediction of model
            defect_mask, heat_map = utils3.postprocess(anomaly_map=prediction["anomaly_map"], img=segmented_tablet, threshold=float(os.getenv('anomaly_mask_threshold')))
            defect_mask=cv2.resize(defect_mask,(blister_img.shape[1],blister_img.shape[0]))
            final_res = utils3.visualize_defect(segmented_tablet, defect_mask)                 
            # cv2.imwrite('/home/frinks3/RISHABH/ACG/save_final_heat_maps'+'/'+str(key)+'defect_mask.png',defect_mask)
            # cv2.imwrite('/home/frinks3/RISHABH/ACG/save_final_heat_maps'+'/'+str(key)+name,heat_map)
            # cv2.imwrite('/home/frinks3/RISHABH/ACG/save_final_visualize_defects'+'/'+str(key)+name,final_res)

            return image,blister_img,defect_mask,blister_dict,blister_coords


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
            # print('blister is empty')
            cv2.rectangle(image,(value[0],value[1]),(value[2],value[3]),(0,0,255),2)
            cv2.putText(image,str(key), (value[0], value[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            # defect_mask=np.zeros((abs(value[3]-value[1]),abs(value[2]-value[0])))
            defect_mask=np.zeros((blister_img.shape[0],blister_img.shape[1]))

            return image,blister_img,defect_mask,blister_dict,blister_coords
            # print(f'blister_dict----{blister_dict}')



############################classification###################################################################

def tablet_classification(image,blister_img,defect_mask,blister_dict,blister_coords,key,value,f,name):

    # for key,value in blister_coords.items():
        # print(f'bisters---dict after anomaly---{blister_dict}')

        # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+'blister.png',blister_img)

        for key1,value1 in blister_dict[key].items():

            # print(key1)
            # print(value1)

            if len(value1)>4:
            
                masked_img=defect_mask[value1[1]:value1[3],value1[0]:value1[2]]

                # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+str(key1)+'mask.png',masked_img)

                tablet_mask=masked_img[value1[5]:value1[7],value1[4]:value1[6]]

                # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+str(key1)+'tablet_mask.png',tablet_mask)
                
                cropped_img=blister_img[value1[1]:value1[3],value1[0]:value1[2]]

                # cv2.imwrite('/home/frinks3/RISHABH/ACG'+'/'+str(key)+str(key1)+'cropped.png',cropped_img)

                print(len(np.where(tablet_mask==255)[0]))

                if len(np.where(tablet_mask==255)[0])>int((os.getenv('tablet_mask_threshold'))):

                    cropped_pil_img= cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    cropped_pil_img=Image.fromarray(cropped_pil_img)

                    defect=predict_image(cropped_pil_img,classification_model,device='cuda')

                    # print(f'name of the defect---{defect}')

                    if defect=='good':
                        f.write(str(str(key)+str(key1)))
                        f.write('\t')
                        print(str(str(key)+str(key1)))
                        print('Some foreign defect')
                        cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,0,255),2)
                        cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                        f.write('Some foreign defect')
                        f.write('\n')

                    else:

                        f.write(str(str(key)+str(key1)))
                        f.write('\t')
                        print(str(str(key)+str(key1)))
                        print(f'{defect} defect')
                        cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,0,255),2)
                        cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                        f.write(f'{defect} defect')
                        f.write('\n')
                else:

                    cropped_pil_img= cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    cropped_pil_img=Image.fromarray(cropped_pil_img)

                    defect=predict_image(cropped_pil_img,classification_model,device='cuda')

                    if defect=='good':
                        f.write(str(str(key)+str(key1)))
                        f.write('\t')
                        print(str(str(key)+str(key1)))
                        cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,255,0),2)
                        cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        print(f'its a good_tablet----{key}{key1}')
                        f.write(f'good_tablet')
                        f.write('\n')

                    else:
                        print(str(str(key)+str(key1)))
                        print(f'{defect} defect')
                        f.write(str(str(key)+str(key1)))
                        f.write('\t')
                        cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,0,255),2)
                        cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                        # print(f'its a good_tablet----{key}{key1}')
                        f.write(f'{defect} defect')
                        f.write('\n')
                    
            else:
                f.write(str(str(key)+str(key1)))
                f.write('\t')
                print(str(str(key)+str(key1)))
                cv2.rectangle(image,(value[0]+value1[0],value[1]+value1[1]),(value[0]+value1[2],value[1]+value1[3]),(0,0,255),2)
                cv2.putText(image,str(str(key)+str(key1)), (value[0]+value1[0], value[1]+value1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                print('Its an empty_pocket')
                f.write('empty_pocket')
                f.write('\n')
                
        return image

####################################################main#########################################################

def main(img_path,f,name):
    
    image = cv2.imread(img_path)
    

    x=time.time()

    blister_dict,blister_coords=pocket_detection(image)

    for key,value in blister_coords.items():
        
        blister_img=image[value[1]:value[3],value[0]:value[2]]

        image,blister_img,defect_mask,blister_dict,blister_coords=anomaly_detection(image,blister_img,blister_dict,blister_coords,key,value,f,name)

        image=tablet_classification(image,blister_img,defect_mask,blister_dict,blister_coords,key,value,f,name)
        
        # cv2.imwrite('/home/frinks3/RISHABH/ACG/'+'result_new.png',image)

    print(f'third----{time.time()-x}')

    return image
    
    # print(f'time----{end-start} seconds')

############################################################################################################


# main('/home/frinks3/RISHABH/ACG/new/testing/Chipping (6).bmp')

img_dir='/home/frinks3/RISHABH/ACG/check'

save_dir='/home/frinks3/RISHABH/ACG/check_save'

with open('save_final_results.txt', 'w') as f:
    for img in os.listdir(img_dir):
        f.write('\n')
        f.write('---------------------------------------------------image------------------------------------')
        f.write('\n')
        f.write(img)
        f.write('\n')
        print(img)
        path=img_dir+'/'+img
        image=main(path,f,img)
        # cv2.imwrite(save_dir+'/'+img,image)









    # img_path='/home/frinks3/RISHABH/ACG/new/Good.png'

