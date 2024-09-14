'''
示例：将图像分割成若干个1280*1280的图像,并为每个小图生成labelme标注文件
python split_image_v2.py --image_file ./dst_yue_images/CE2_GRAS_DOM_G019_07N153E_A.jpg --label_file ./dst_yue_images/G019_new.json --winsize 1280 --overlap_width 250 --output_dir ./output
'''

import cv2
import os
import os.path as osp
import argparse
import traceback

import copy

import json
import base64

labelme_content = {
  "version": "4.5.6",
  "flags": {},
  "shapes":[],
  "imagePath": "20191231_S162.jpg",
  "imageData":"", 
  "imageHeight": 2736,
  "imageWidth": 3648
}

def get_bbox(json_file):
    with open(json_file, 'r',encoding='utf-8') as json_file:
        data = json.load(json_file)
    # shapes = data["shapes"]
    # # print(data)
    # for shape in shapes:

    features = data["features"]
    bboxs = []
    for feature in features:
        # print(feature)
        try:
            x = feature["attributes"]["x_pixel"]
            y = feature["attributes"]['y_pixel']
            d = feature["attributes"]['Diameter']
            r = d/7/2  # 7m分辨率的图像
            lt_x = int(x-r)
            lt_y = int(y-r)
            rd_x = int(x+r)
            rd_y = int(y+r)
            # print((lt_x,lt_y,rd_x,rd_y))
            bboxs.append((lt_x,lt_y,rd_x,rd_y))
        except Exception as e:
            #print("exception:",str(e))
            pass
    # print(bboxs)
    return bboxs

def get_bbox_csv(csv_file):
    import csv
    # 打开CSV文件并读取内容  
    with open(csv_file, mode='r', newline='') as csv_file:  
        csv_reader = csv.DictReader(csv_file)  # 使用DictReader可以直接通过列名访问数据  
        bboxs = []  # 用于存储gt_x1, gt_y1, gt_x2, gt_y2的值  
        for row in csv_reader:  
            print("++++++:",row)
            lt_x = int(row['gt_x1'].split(".")[0])
            lt_y = int(row['gt_y1'].split(".")[0])
            rd_x = int(row['gt_x2'].split(".")[0])
            rd_y = int(row['gt_y2'].split(".")[0])
            if lt_x == rd_x:
                continue

            bboxs.append((lt_x,lt_y,rd_x,rd_y))
        return bboxs
    

def box_in_box(in_box,out_box):
    if in_box[0] >= out_box[0] and in_box[1] >= out_box[1] \
       and in_box[2]<=out_box[2] and in_box[3] <= out_box[3] :
       return True
    else:
        return False


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


def run(
        image_file="",
        label_file="",

        winsize = 1280,
        overlap_width = 250,
        output_dir = "./"):
    
    # image_file = "/DATA01/yue_7m_jpg_black/CE2_GRAS_DOM_07m_N021_85N000W_A.jpg"
    # label_file = "/DATA01/yue_json/N021_new.json"
    # output_dir = "/DATA01/yolov5-7.0-train/jpg_7m_split_labelme/N021"
    
    if not os.path.exists(image_file):
        print("image file path is not exist ")
        raise Exception('image file path is not exist')
    if not os.path.exists(output_dir):
        # print("output_dir path is not exist ")
        os.mkdir(output_dir)
        # raise Exception('output_dir path is not exist')

    try:
        image = cv2.imread(image_file)
        h,w,_ = image.shape
        print(h,w)

        # 解析label_file，从中读取所有标注框
        #bboxs = get_bbox(label_file)
        bboxs = get_bbox_csv(label_file)
        print("bboxes num:",len(bboxs))

        prefix = os.path.basename(image_file).split('.')[0]
        step = winsize
        overlap = overlap_width
        step_real = step - overlap
        window_w = winsize
        window_h = winsize
        offset_x = 0
        offset_y = 0

        iter_x = int((w-step_real+1)/step_real)
        iter_y = int((h-step_real+1)/step_real)
        for id_y in range(iter_y):
            for id_x in range(iter_x):
                offset_x = step_real*id_x
                offset_y = step_real*id_y
                right_x = offset_x + window_w
                right_y = offset_y + window_h
                right_x = min(right_x,w-1)
                right_y = min(right_y,h-1)
                
                roi_box = [offset_x,offset_y,right_x,right_y]
                roi_img = image[offset_y:right_y,offset_x:right_x]
                
                # 遍历所有box，找到所有在roi_img中的box
                dst_bboxs = []
                for box in bboxs:
                    if box_in_box(box,roi_box) == True:
                        new_box = [box[0]-offset_x,box[1]-offset_y,box[2]-offset_x,box[3]-offset_y]
                        dst_bboxs.append(new_box)
                if len(dst_bboxs)==0:
                    continue
                print("dst_bboxes size:",len(dst_bboxs))
                # 抠出图像
                filename = prefix + "_" + str(id_x) + "_" + str(id_y) +".jpg"
                filename = os.path.join(output_dir,filename)
                cv2.imwrite(filename,roi_img)
                # 生成与图像对应的标注文件
                new_data = copy.deepcopy(labelme_content)
                base64_img = load_image_base64(filename)
                new_data["imagePath"]= filename.split("/")[-1]
                new_data["imageData"] = base64_img
                tt_shapes = []
                for tbox in dst_bboxs:
                    item = generate_one_label("keng",tbox)
                    tt_shapes.append(item)
                new_data["shapes"] = tt_shapes

                json_file = prefix + "_" + str(id_x) + "_" + str(id_y) +".json"
                json_file = os.path.join(output_dir,json_file)
                label_json_str = json.dumps(new_data)
                with open(json_file,'w') as f:
                    f.write(label_json_str)
                print(json_file)


    except Exception as e:
        traceback.print_exc()

    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str,help='图片路径')
    parser.add_argument('--label_file', type=str,help='标注文件路径')

    parser.add_argument('--winsize', type=int,  default=1280,help='分割后子图的尺寸')
    parser.add_argument('--overlap_width', type=int , default=250 ,help='子图间重叠宽度')
    parser.add_argument('--output_dir', type=str , default="./" ,help='输出目录')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)