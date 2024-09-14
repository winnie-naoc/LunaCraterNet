import os
import numpy as np
import json
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from os import getcwd
 
classes = ["keng"]
# 1.标签路径
labelme_path = "yueqiu/"
isUseTest = True  # 是否创建test集
# 3.获取待处理文件
files = glob(labelme_path + "*.json")
files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
print(files)
if isUseTest:
    trainval_files, test_files = train_test_split(files, test_size=0.1, random_state=55)
else:
    trainval_files = files
# split
train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=55)
 
 
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
 
 
wd = getcwd()
print(wd)
 
 
def ChangeToYolo5(files, txt_Name):
    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')
    list_file = open('tmp/%s.txt' % (txt_Name), 'w')
    for json_file_ in files:
        json_filename = labelme_path + json_file_ + ".json"
        imagePath = labelme_path + json_file_ + ".jpg"
        list_file.write('%s/%s\n' % (wd, imagePath))
        out_file = open('%s/%s.txt' % (labelme_path, json_file_), 'w')
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread(labelme_path + json_file_ + ".jpg").shape
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0]) if min(points[:, 0]) > 0 else 0
            xmax = max(points[:, 0]) if max(points[:, 0]) > 0 else 0
            ymin = min(points[:, 1]) if min(points[:, 1]) > 0 else 0
            ymax = max(points[:, 1]) if max(points[:, 1]) > 0 else 0
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                cls_id = classes.index(label)
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert((width, height), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                print(json_filename, xmin, ymin, xmax, ymax, cls_id)
 
 
ChangeToYolo5(train_files, "train")
ChangeToYolo5(val_files, "val")
ChangeToYolo5(test_files, "test")
