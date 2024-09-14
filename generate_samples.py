'''

1. 准备好使用labelme标注的样本图像，放到目录yueqiu_template
2. 准备制作好的背景图像，放到目录bg中
3. 执行：python3 generate_samples.py ，合成的样本保存在dst_yue_images
'''
import cv2 as cv
import json
import os
import base64
import copy
import numpy as np
import time


bg_dir = "/DATA02/0628_dataset/bg_0628"               # 背景图像目录
labeled_dir = "/DATA02/0628_dataset/label_0628"       # 已标注的样本目录
generated_dir = "/DATA02/0628_dataset/generate_0628"  # 生成样本目录


generated_type = ""  # 生成样式

if not os.path.exists(generated_dir):
    os.makedirs(generated_dir)
    
def load_image_base64(img_path: str):
    """
    Loads an encoded image as an array of bytes.
    
    """
    with open(img_path, "rb") as f:
        #img = base64.b64encode(f.read())
        img = base64.b64encode(f.read()).decode('utf-8')
        return img
    
def generate_one_label(label,points):
    item =  {
                "label":label,
                "points":[
                    [
                        points[0],
                        points[1]
                    ],
                    [
                        points[2],
                        points[3]
                    ]
                ],
                "group_id":0,
                "shape_type":"rectangle",
                "flags":{}
            }
    return item


# 
def apply_replace_background(srcimg,backimg,src_bboxs,scale_ratio):
    dst_bboxs = []
    for bbox in src_bboxs:
        tt = srcimg[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        dst_bbox = [int(p*scale_ratio) for p in bbox]
        dst = cv.resize(tt,(dst_bbox[2]-dst_bbox[0],dst_bbox[3]-dst_bbox[1]))
        backimg[dst_bbox[1]:dst_bbox[3],dst_bbox[0]:dst_bbox[2]] = dst
        dst_bboxs.append(dst_bbox)
    return dst_bboxs

def zoom_out_bbox(bbox,h,w,tile_w):
    bbox[0] = max(bbox[0]-tile_w,0)
    bbox[1] = max(bbox[1]-tile_w,0)
    bbox[2] = min(bbox[2]+tile_w,w)
    bbox[3] = min(bbox[3]+tile_w,h)
    return bbox

def apply_replace_background_v2(srcimg,backimg,src_bboxs,scale_ratio):
    dst_bboxs = []
    for bbox in src_bboxs:
        dst_bbox = [int(p*scale_ratio) for p in bbox]   # 记录bbox在目标图像中的位置
        t_bbox = copy.deepcopy(bbox)
        big_box = zoom_out_bbox(t_bbox,1280,1280,30)    
        tt = srcimg[big_box[1]:big_box[3],big_box[0]:big_box[2]]  # 从原图中抠出来比标注框大30个像素的区域
        big_box = [int(p*scale_ratio) for p in big_box]
        dst = cv.resize(tt,(big_box[2]-big_box[0],big_box[3]-big_box[1]))
        backimg[big_box[1]:big_box[3],big_box[0]:big_box[2]] = dst
        dst_bboxs.append(dst_bbox)
    return dst_bboxs

def apply_replace_background_v3(srcimg,backimg,src_bboxs,scale_ratio):
    dst_bboxs = []
    for bbox in src_bboxs:
        if abs(bbox[2]-bbox[0]) > 600 or abs(bbox[3]-bbox[1]) > 600:
            continue
        if abs(bbox[2]-bbox[0]) <15 or abs(bbox[3]-bbox[1]) < 15:
            continue
        dst_bbox = [int(p*scale_ratio) for p in bbox]   # 记录bbox在目标图像中的位置
        t_bbox = copy.deepcopy(bbox)
        big_box = zoom_out_bbox(t_bbox,1280,1280,40)    
        tt = srcimg[big_box[1]:big_box[3],big_box[0]:big_box[2]]  # 从原图中抠出来比标注框大30个像素的区域
        big_box = [int(p*scale_ratio) for p in big_box]
        dst = cv.resize(tt,(big_box[2]-big_box[0],big_box[3]-big_box[1]))
        # 随机的将图像进行水平或者垂直翻转
        # todo

        mask = 255*np.ones(dst.shape, dst.dtype)
        center = ((big_box[0]+big_box[2])//2,(big_box[1]+big_box[3])//2)
        #backimg[big_box[1]:big_box[3],big_box[0]:big_box[2]] = dst
        backimg = cv.seamlessClone(dst, backimg, mask, center, cv.MIXED_CLONE)
        dst_bboxs.append(dst_bbox)
    return dst_bboxs,backimg


# 1. 生成一组背景模板
backimg_files = os.listdir(bg_dir)
backimg_arr = []    #背景图像列表
for backimg_file in backimg_files:
    backimg_file = os.path.join(bg_dir,backimg_file)
    timg = cv.imread(backimg_file)
    backimg = cv.resize(timg,(1280,1280))
    backimg_arr.append(backimg)
    # 
    # new_backimg_file = backimg_file.replace("back","new_back")
    # cv.imwrite(new_backimg_file,backimg)

candidata_dir_list = ["images"]
data_dd = "_628_"

for candidata_dir in candidata_dir_list:
    labeled_dir_2 = os.path.join(labeled_dir,candidata_dir)
    generated_dir_2 = os.path.join(generated_dir,candidata_dir)
    if not os.path.exists(generated_dir_2):
        os.makedirs(generated_dir_2)
    # 2. 遍历已标注数据，将标注框区域内容拷贝到背景模板中并保存
    # scale_ratio_arr = [1.0,0.8,0.6,0.7,0.5]   # 四个缩放尺寸
    # scale_ratio_arr = [1.0,0.8,0.6,0.5,0.4]   # 四个缩放尺寸
    scale_ratio_arr = [1.0,0.8]   # 四个缩放尺寸
    label_files = os.listdir(labeled_dir_2)
    labeljson_files = [os.path.join(labeled_dir_2,fn) for fn in label_files if fn.endswith(".json")]
    for labeljson in labeljson_files:
        with open(labeljson,'r',encoding='UTF-8') as f:
            data = json.loads(f.read())
            # 1.读取对应的图像
            # img_path = data['imagePath']
            prefix_name = labeljson.split("/")[-1].split(".")[0] + data_dd
            # img_path = os.path.join(labeled_dir,img_path)
            img_path = labeljson
            img_path = img_path.replace("json","png")# png或者jpg
            if not os.path.exists(img_path):
                img_path = img_path.replace("png","jpg")
            curr_img = cv.imread(img_path)

            # 2.读取所有的标注框
            shapes = data["shapes"]
            bboxs = []
            for item in shapes:
                points = item["points"]
                bbox = [int(points[0][0]),int(points[0][1]),int(points[1][0]),int(points[1][1])]
                bbox[0]=max(0,bbox[0])
                bbox[1]=max(0,bbox[1])
                bbox[2]=max(0,bbox[2])
                bbox[3]=max(0,bbox[3])
                if bbox[0]>bbox[2]:
                    bbox[0],bbox[2]=bbox[2],bbox[0]
                if bbox[1]>bbox[3]:
                    bbox[1],bbox[3]=bbox[3],bbox[1]
                bboxs.append(bbox)
            
            #3.
            for i,tback in enumerate(backimg_arr): # 遍历背景图像
                for j,scale in enumerate(scale_ratio_arr): # 遍历缩放尺寸
                    tt_back = tback.copy()
                    tt_boxs = copy.deepcopy(bboxs)
            
                    dst_bboxs,tt_back = apply_replace_background_v3(curr_img,tt_back,tt_boxs,scale)
                    #dst_bboxs = apply_replace_background_v2(curr_img,tt_back,tt_boxs,scale)
                    
                    # 保存生成的图片
                    img_filename = prefix_name + "_" + str(i)+"_" + str(j) + ".png"
                    generate_img = os.path.join(generated_dir_2,img_filename)
                    cv.imwrite(generate_img,tt_back)
                    print(img_filename)
                    # 重新生成标注文件
                    new_data = copy.deepcopy(data)
                    base64_img = load_image_base64(generate_img)
                    new_data["imagePath"]= generate_img.split("/")[-1]
                    new_data["imageData"] = base64_img
                    tt_shapes = []
                    for tbox in dst_bboxs:
                        item = generate_one_label("keng",tbox)
                        tt_shapes.append(item)
                    new_data["shapes"] = tt_shapes

                    json_file = generate_img.split('.')[0] + ".json"
                    label_json_str = json.dumps(new_data)
                    with open(json_file,'w') as f:
                        f.write(label_json_str)






        




