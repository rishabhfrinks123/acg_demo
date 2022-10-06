import numpy as np
import cv2
import os
import torch

class Segmentator:
    
    def __init__(self):
        pass
        
    # Function that passes the image to the model and get the segmentation output
    def segment(self, model, img, threshold_value=0.5, device="cuda"):
        a_img = img.copy()
        shape = img.shape[:-1]
        img = cv2.resize(img, (512,512))
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = img[np.newaxis, ...]
        # img=img/255
        img = torch.from_numpy(img)
        if device == "cuda":
            img = img.permute([0, 3, 1, 2]).to("cuda")
        else:
            img = img.permute([0, 3, 1, 2])
        with torch.no_grad():
            model.eval()
            logits = model(img.float())
        if device == "cuda":
            pr_masks = logits.sigmoid().cpu()
        else:
            pr_masks = logits.sigmoid()
        mask = pr_masks[0].permute([1,2,0])
        mask = cv2.resize(np.array(mask), (shape[1], shape[0]))
        (T, mask) = cv2.threshold(mask,0.5, 255, cv2.THRESH_BINARY)
        a_img[mask==0]=0
        # mask = mask[..., np.newaxis]
        # mask = mask.astype("uint8")
        # contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return a_img,mask

    # def boost_contrast(self, img):
    #     lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #     l_channel, a, b = cv2.split(lab)
    #     # Applying CLAHE to L-channel
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #     cl = clahe.apply(l_channel)
    #     # mergnig
    #     limg = cv2.merge((cl,a,b))
    #     # Converting to bgr
    #     enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #     return enhanced_img

    # Function that returns output for part segmentations
    # def parts_segment(self, model, img, threshold, device="cuda"):
    #     a_img = self.segment( model, img, threshold, device="cuda")
    #     contours = sorted(contours, key=len, reverse=True)
    #     new_mask = np.zeros(mask.shape)
    #     final = cv2.fillPoly(new_mask, pts=[contours[0]], color=(255, 255, 255))
    #     a_img[final[:,:,0]==0] = 0
    #     return contours, a_img, final



