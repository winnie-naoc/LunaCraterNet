# LunaCraterNet
We propose a novel method using the YOLOv5l deep learning model to detect small craters on the Moon's surface. By leveraging a high-quality, diverse dataset with data augmentation and optimization, we enhanced the model's detection performance. This document explains the procedures used in the detection process.

## Install
```shell
pip install -r requirements.txt
```
,,,
Hardware requirements: nvidia GPU(memory > 12GB), 8GB RAM
Program language: python
Software required: Pytorch, torchvision, numpy, PIL, labelme, onnx, onnxruntime,  opencv-python
Program size: 152MB
,,,


## Sample Synthesis and Generation
The synthesis and generation of crater samples is a key feature of this study. Unlike the traditional method of manual annotation, this project builds upon manual labeling and employs image processing algorithms to synthesize a large number of crater samples. This significantly expands the size of the training dataset and enhances the model's generalization ability. The detailed sample generation process is as follows:
1. Preparation of Original Images
Select and download the original lunar map subdivisions from the following link: https://doi.org/10.12350/CLPDS.GRAS.CE2.DOM-7m.vA. After downloading, convert the images from TIFF format to JPG format while maintaining the original resolution.

2. Image Segmentation
The lunar map subdivisions are divided into several 1280x1280 sized images.
```shell
python3 split_image.py --image_file small.jpg --winsize 128 --overlap_width 25 --output_dir ./output
```
3. Crater Annotation
Select no fewer than 500 images from the segmented images and annotate them using Labelme. The selected images should contain as many different types of craters as possible, and the annotated craters must have very distinct features. When annotating with Labelme, only mark craters with sizes between 25 and 300 pixels, ensuring that only craters with clear and prominent features are annotated. It is not necessary to annotate every crater in each image.

4. Background Image Creation
Images with fewer craters are selected from the segmented smaller images. The craters and potential craters in these images are then filled and masked using surrounding background elements.
Prepare at least two background images for each type of region.

5. Image Synthesis
```shell
# 1) Run `generate_samples.py` to synthesize new images, using the details provided in the comment section of the reference file.  
# 2) Run `labelmeToYolov5.py` to convert the Labelme annotations into a format compatible with YOLOv5. This will generate some files in the `tmp` directory within the current folder. For details, refer to the comments in `labelmeToYolov5.py`.  
# 3) Navigate to the `tmp` directory and execute `python makedata.py`. This will generate a `VOC` directory in the parent folder.
```

## Training
```shell
python3  -m torch.distributed.launch --nproc_per_node 2 train.py --img 1280 --batch 8 --epoch 20  --data data/keng.yaml --cfg models/keng_yolov5l.yaml --weights weights/yolov5l6.pt --hyp data/hyps/hyp.keng.yaml
```

## Model Validation
```shell
python3 detect.py --imgsz 1280 --weights /path/weights/best.pt --source VOC/images/test/

```

# Model Conversion
python3 export.py --weights /path/weights/best.pt --imgsz 1280 --include onnx
python inference.py --images-dir="/path/to/images" --target-dir="/path/to/target" --onnx-model-path="/path/to/model.onnx" --conf-thre=0.25
```
