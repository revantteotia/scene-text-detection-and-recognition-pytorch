# import sys
# sys.path.append('CRAFT')
# sys.path.append('deepTextRecognition')

import json
from CRAFT import extract_OCR_bbox
from deepTextRecognition import recognize_text
import combine_ocr_bbox


if __name__ == "__main__":
    
    img_folder_path = 'data'
    cropped_text_out_dir = 'result'
    trained_model_path = 'pretrained-models/text_detection_pretrained_model/craft_mlt_25k.pth'
    
    # extracts cropped scene text in './result' folder
    extract_OCR_bbox.extract_OCR_bbox(img_folder_path, trained_model_path)

    # now recognizing texts from cropped scene texts in 'results' directory

    recognition_saved_model = 'pretrained-models/text-recognition/TPS-ResNet-BiLSTM-Attn.pth'
    recognize_text.recognize_text(recognition_saved_model, cropped_text_out_dir)

    detected_scene_text_json_path = 'ocr_detection.json'

    ocr_with_bbox = combine_ocr_bbox.combine_ocr_bbox(cropped_text_out_dir, detected_scene_text_json_path)

    ocr_with_bbox_json_path = 'ocr_with_bbox.json'

    with open(ocr_with_bbox_json_path, 'w') as fp:
        json.dump(ocr_with_bbox, fp)
