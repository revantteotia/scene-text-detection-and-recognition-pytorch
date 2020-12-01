# updated file_utils.py in CRAFT
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from CRAFT import imgproc

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def imcrop(img, bbox, pad_img=False): 
    x1,y1,x2,y2 = bbox
    
    if pad_img:
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    
    else:
        x1 = min(img.shape[1], max(0, x1))
        x2 = min(img.shape[1], max(0, x2))
        y1 = min(img.shape[0], max(0, y1))
        
        y2 = min(img.shape[0], max(0, y2))
    return img[y1:y2, x1:x2, :], (x1, y1, x2, y2)

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2

def saveCroppedOCRResult(img_file, img, boxes, dirname='./output/', verticals=None, texts=None):
    """ save text detection result one by one
    Args:
        img_file (str): image file name
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        None
    """
    img = np.array(img)
    ###########################
    original_img = np.copy(img)
    ###########################

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # result directory
    res_file = os.path.join(dirname, filename + '.csv')
    res_img_file = os.path.join(dirname, filename + '.jpg')

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # making cropped images directory : #######################
    cropped_images_dir = os.path.join(dirname,  filename)
    
    if not os.path.isdir(cropped_images_dir):
        os.mkdir(cropped_images_dir)
    ###########################################################

    

    with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            # saving cropped image
            #################################################
            cropped_img_path = cropped_images_dir + '/' + str(i) + '.jpg' 
            print("cropped_img_path : ", cropped_img_path)
            print("poly = , ", poly)

            x1 = min(poly[0], poly[2], poly[4], poly[6]) 
            x2 = max(poly[0], poly[2], poly[4], poly[6])
            y1 = min(poly[1], poly[3], poly[5], poly[7])
            y2 = max(poly[1], poly[3], poly[5], poly[7])

            # y_bot = min(poly[0], poly[2], poly[4], poly[6]) 
            # y_top = max(poly[0], poly[2], poly[4], poly[6])
            # x_left = min(poly[1], poly[3], poly[5], poly[7])
            # x_right = max(poly[1], poly[3], poly[5], poly[7])

            cropped_image, (x1, y1, x2, y2) = imcrop(original_img, (x1, y1, x2, y2))
            # bbox = (x1, y1, x2, y2)
            norm_bbox = (x1/img.shape[1], y1/img.shape[0], x2/img.shape[1], y2/img.shape[0])
            print('text bbox: ', (x1, y1, x2, y2))
            print('text normalized bbox: ', norm_bbox)
            cv2.imwrite(cropped_img_path, cropped_image)


            # cv2.imwrite(cropped_img_path, original_img[x_left:x_right, y_bot:y_top ])
            ################################################

            # strResult = ','.join([str(p) for p in poly]) + '\r\n'
            # saving normalized bbox
        
            strResult = str(i) + '.jpg, ' + ','.join([str(b) for b in norm_bbox]) + '\r\n' 

            f.write(strResult)

            # poly = poly.reshape(-1, 2)
            # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            # ptColor = (0, 255, 255)
            # if verticals is not None:
            #     if verticals[i]:
            #         ptColor = (255, 0, 0)

            # if texts is not None:
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     font_scale = 0.5
            #     cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
            #     cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

    # Save result image
    # cv2.imwrite(res_img_file, img)

