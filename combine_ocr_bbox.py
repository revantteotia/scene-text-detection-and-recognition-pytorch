import pandas as pd
import glob
import os

import json


def combine_ocr_bbox(bbox_info_dir_path, ocr_detection_json_path):
    '''
    returns a dict with img_filename as keys
    '''
    # detection_df = pd.read_csv(orc_token_csv_path, header=None, index_col=0)

    with open(ocr_detection_json_path) as fp:
        ocr_detection_dict = json.load(fp)

    # its variable name is list but this is a dict
    ocr_info_list = {}

    path = bbox_info_dir_path + "/*.csv"
    for fname in glob.glob(path):
        ocr_info = []
        reco_df = pd.DataFrame() # empty df 
        # print('image name= ',fname)
        # print(reco_df.head())
        try :
            reco_df = pd.read_csv(fname, header=None, index_col=0)
        except Exception as e: 
            print(e)
            

        cropped_texts_directory_path = fname.split('.')[0]
        filebasename = cropped_texts_directory_path.split('/')[-1]
        # print('cropped_texts_directory_path = ', cropped_texts_directory_path)

        for index, row in reco_df.iterrows():
            ocr_token_info = {}
            # print('crop name = ', index)
            x1, y1, x2, y2 = row[1], row[2], row[3], row[4]
            norm_bbox = (x1, y1, x2, y2)
            # print('norm bbox= ', norm_bbox)

            text_detection_result_index = cropped_texts_directory_path + '/' + index
            detected_text = ocr_detection_dict[text_detection_result_index]['word']
            detected_text_confidence = ocr_detection_dict[text_detection_result_index]['confidence_score']

            # print ('text_detection_result_index = ', text_detection_result_index)
            # print ('detected_text = ', detected_text)
            # print ('detected_text_confidence = ', detected_text_confidence)

            ocr_token_info['word'] = detected_text.strip()
            ocr_token_info['norm_bbox'] = norm_bbox
            ocr_token_info['confidence'] = detected_text_confidence
            ocr_info.append(ocr_token_info)
            
        ocr_info_list[filebasename] = ocr_info
    
    return ocr_info_list

if __name__ == "__main__":
    bbox_info_dir_path = 'result'
    orc_token_csv_path = 'detection_csv.csv'
    ocr_detection_json_path = 'ocr_detection.json'

    ocr_info_list = combine_ocr_bbox(bbox_info_dir_path, ocr_detection_json_path)

    print(len(ocr_info_list))
    print(ocr_info_list.keys())
    print(list(ocr_info_list.items())[0])

