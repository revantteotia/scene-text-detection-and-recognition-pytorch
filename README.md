# scene-text-detection-and-recognition-pytorch
Combining separate scene text detection and recognition modules into one : To get scene-text in images with bbox

It uses CLOVA.AI's detector and recognizer

Detector : https://github.com/clovaai/CRAFT-pytorch \
Recognizer : https://github.com/clovaai/deep-text-recognition-benchmark

[Pre-trained detector model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) ('craft_mlt_25k.pth')\
[Pre-trained Recognizer model](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW) ('TPS-ResNet-BiLSTM-Attn.pth')\

1. Put images in data directory
2. data directory structure should be like :

```
data
+-- image_class_0
|   +-- image_class_0_0.jpg
|   +-- image_class_0_1.jpg
|   +-- image_class_0_2.jpg
+-- image_class_1
|   +-- image_class_1_0.jpg
|   +-- image_class_1_1.jpg
|   +-- image_class_1_2.jpg
```

3. Put pretrained text-detector model at 'pretrained-models/text_detection_pretrained_model/craft_mlt_25k.pth'
4. Put pretrained text-recognition model at 'pretrained-models/text-recognition/TPS-ResNet-BiLSTM-Attn.pth'
5. run `$python text_detect_and_recognize.py`
6. 'ocr_with_bbox.json' will have detected text with normalized bbox and confidence score


### TODO
1. Clean the code
2. LICENSE ?