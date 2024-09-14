'''
示例：将图像分割成若干个1280*1280的图像
python split_image_dt.py --image_file test.jpg --winsize 1280 --overlap_width 250 --output_dir ./output
'''
import cv2
import os
import os.path as osp
import argparse
import traceback

import copy

def run(
        image_file="",
        winsize = 1280,
        overlap_width = 250,
        output_dir = "./"):
    
    if not os.path.exists(image_file):
        print("image file path is not exist ")
        raise Exception('image file path is not exist')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        # print("output_dir path is not exist ")
        # raise Exception('output_dir path is not exist')

    try:
        image = cv2.imread(image_file)
        h,w,_ = image.shape

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
                roi_img = image[offset_y:right_y,offset_x:right_x]

                filename = prefix + "_" + str(id_x) + "_" + str(id_y) +".jpg"
                print(filename)
                filename = os.path.join(output_dir,filename)
                cv2.imwrite(filename,roi_img)
    except Exception as e:
        traceback.print_exc()

    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str,help='图片目录')
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