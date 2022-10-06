import numpy as np
import cv2

# def rotate_tablets(image):
#     img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     black_img = np.zeros((image.shape[0],image.shape[1],3), np.uint8)
#     cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  

#     # print(len(cnts))

#     c = max(cnts, key=cv2.contourArea)


#     ellipse = cv2.fitEllipse(c)

#     (h, w) =image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)

#     M = cv2.getRotationMatrix2D((cX, cY),ellipse[2], 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h))

#     rotate_gray=cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

#     cnts = cv2.findContours(rotate_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  

#     c = max(cnts, key=cv2.contourArea)

#     extLeft = tuple(c[c[:, :, 0].argmin()][0])
#     extRight = tuple(c[c[:, :, 0].argmax()][0])
#     extTop = tuple(c[c[:, :, 1].argmin()][0])
#     extBot = tuple(c[c[:, :, 1].argmax()][0])

#     crop_img=rotated[extTop[1]:extBot[1],extLeft[0]:extRight[0]]

#     centre_crop=(crop_img.shape[1]//2,crop_img.shape[0]//2)

#     centre_black=(black_img.shape[1]// 2,black_img.shape[0]// 2)

#     i=int(centre_black[0]-centre_crop[0])
#     j=int(centre_black[1]-centre_crop[1])

#     black_img[j:j+crop_img.shape[0],i:i+crop_img.shape[1]]=crop_img

#     return black_img


def rotate_tablets(image,blister,key,key1):

    img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_img = np.zeros((512,512,3), np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  

    # print(len(cnts))

    if len(cnts)>0:

        c = max(cnts, key=cv2.contourArea)


        ellipse = cv2.fitEllipse(c)

        # image = cv2.ellipse(image,ellipse,(0,255,0),2)

        (h, w) =image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY),ellipse[2], 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        rotate_gray=cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(rotate_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2] 

        if len(cnts)>0:

            c = max(cnts, key=cv2.contourArea)

            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            crop_img=rotated[extTop[1]:extBot[1],extLeft[0]:extRight[0]]

            # centre_crop=(crop_img.shape[1]//2,crop_img.shape[0]//2)

            # centre_black=(black_img.shape[1]// 2,black_img.shape[0]// 2)

            # i=int(centre_black[0]-centre_crop[0])
            # j=int(centre_black[1]-centre_crop[1])

            # black_img[j:j+crop_img.shape[0],i:i+crop_img.shape[1]]=crop_img

            blister[key][int(key1)]=blister[key][int(key1)]+[int(extLeft[0]),int(extTop[1]),int(extRight[0]),int(extBot[1])]

        else:
            pass
    
    else:
        pass

        # print(blister)

    # blister=[j:j+crop_img.shape[0],i:i+crop_img.shape[1]]

    return crop_img


# black_img = np.zeros((786,1280,3), np.uint8)
# cv2.imwrite('/home/frinks3/RISHABH/ACG/test_batch_10/blac_img.png',black_img)
