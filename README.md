月球撞击坑检测模型训练说明：
1. 制作样本。
（1）将月球图像先分割成1280*1280分辨率的图像。图像间重叠区域宽度为280像素（如果不重叠的话，可能一些处在两幅图边界上的坑将无法检测到。）
 ```shell
 python split_image_dt.py --image_file test.jpg --winsize 1280 --overlap_width 250 --output_dir ./output_dir
 ```
（2）对（1）中分割后的小图进行标注。可以选择任何标注工具，本教程中提供的脚本是基于labelme标注的，如果使用其他标注工具标注的样本，可以自行先转为labelme的格式。标注图像时应尽可能覆盖不同尺寸、不同类型、不同光线、不同背景的目标坑。
（3）制作背景图像，背景图像应尽可能覆盖月球不同的地貌特征。
（4）样本制作1. 
```shell
#  假设labelme标注的样本目录为/DATA01/show_resources/labelme_files/
#  假设背景图像目录为/DATA01/show_resources/background_images/
#  假设当前程序目录为：/DATA01/yolov11/
#（1）修改generate_samples.py文件中的bg_dir,labeled_dir,generated_dir,date_dd四个参数
      bg_dir = "/DATA01/show_resources/background_images"               # 背景图像目录
      labeled_dir = "/DATA01/show_resources/labelme_files"              # 已标注的样本目录
      generated_dir = "/DATA01/show_resources/generated_1209"           # 生成样本目录
      date_dd = "_1209_"                                                # 时间标签（会体现在文件名上）
      然后执行python generate_samples.py,生成样本。
      执行完成后，将在generated_dir目录下生成新的样本。
# （2）修改labelmeToYolov5.py文件中的labelme_path参数，修改为步骤（1）中的generated_dir目录+"images/"
     labelme_path = "/DATA01/show_resources/generated_1209/images/"    # labelme标注文件路径, 注意最后一个字符必须是/
     执行完成后将在当前目录下新生成一个tmp目录。
#  (3) 修改makedata.py并执行，生成训练样本。修改参数如下：
     base_voc_dir = './train_data_1209'   # 最终生成的训练样本目录
     base_tmp_dir = './tmp'               # 步骤（2）中生成的tmp目录
     执行完成后，将在base_voc_dir目录下生成新的训练样本。即为/DATA01/yolov11/train_data_1209
```
 

2. 训练
（1）设置训练参数。/DATA01/yolov11/ultralytics/cfg/models/11/yolo11.yaml 修改nc为1
（2）设置样本路径。/DATA01/yolov11/ultralytics/cfg/datasets/keng.yaml 修改path为步骤1中生成的样本目录/DATA01/yolov11/train_data_1209
（3）执行训练。python train.py,训练完成后，将在runs/train下生成一个新的exp为前缀的新文件目录，该目录下包含训练好的模型文件。
 (4) 执行测试。命令为：yolo task=detect mode=predict model=/DATA01/yolov11/runs/train/exp2/weights/best.pt source=/DATA01/show_resources/labelme_files/images/CE2_GRAS_DOM_C104_66N127W_A_3_3.jpg  ，其中model参数为步骤3中生成的best.pt文件。source参数为待检测的图像。检测结果将保存在/DATA01/yolov11/runs/detect/predict/下。

 3. 模型导出为onnx格式。方便部署
 ```shell
 # 导出onnx格式
 yolo export model=/DATA01/yolov11/runs/train/exp2/weights/best.pt --format=onnx --opset=13 --simplify
 #导出后的onnx模型保存在/DATA01/yolov11/runs/exp2/weights/下
 # 验证onnx模型
 python detect.py --model=/DATA01/yolov11/runs/train/exp2/weights/best.onnx --image=/DATA01/show_resources/labelme_files/images/CE2_GRAS_DOM_C104_66N127W_A_3_3.jpg --output=/DATA01/yolov11/test_images/output.jpg
 ```

 4. 月球撞击坑检测
 修改inference.py中的如下参数：
 ```python
    images_dir = "/DATA02/CE2_7m_JPG/C区/"                 # 待检测的图像目录
    target_dir = "/DATA02/CE2_7m_JPG/results/C区/"         # 检测结果保存目录
    onnx_model_path = "/DATA01/yolov11/runs/train/exp2/weights/best.onnx"  # 模型路径
 ```
 然后执行`python inference.py`即可，执行完成后将在target_dir目录下生成检测结果。


