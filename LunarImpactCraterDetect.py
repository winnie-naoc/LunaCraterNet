'''
撞击坑检测流程：

1. 读取一张图片
2. 由于原图非常大,所以需要将原图裁剪为若干个1280*1280像素的图像块然后进行撞击坑检测。
   为避免因图像裁剪时部分撞击坑因处在裁剪区域边界而导致的漏检问题，后一个裁剪区域将于上一个裁剪区域重叠
   一个像素20米, 那么直径尺寸处在500-5000m撞击坑在图像上的像素宽度处在25-250像素。
3. 遍历每一个裁剪的子区域，执行检测
'''

import os
import cv2 as cv
import numpy as np
import onnxruntime

base_dir = os.path.dirname(os.path.abspath(__file__))


class Detector(object):
    def __init__(self) -> None:
        self.model = 



if __name__ == '__main__':

    image_file = os.path.join("/data1/dengtao/develop_dt/yueqiu_test/CE2_GRAS_DOM_C001_63N165W_A.jpg")
    ori_img = cv.imread(image_file)
    img_h,img_w= ori_img.shape[:-1]

    win_size = 1280
    overlap_size = 280
    step = 1000
    iter_rows = (img_h + step - 1)//step
    iter_cols = (img_w + step - 1)//step
    print("iter_rows*iter_cols:{}*{}={} ",iter_rows,iter_cols,iter_rows*iter_cols)

    # for row in range(iter_rows):
    #     for col in range(iter_cols):
    #         img = ori_img[row*step:row*step+win_size,col*step:col*step + win_size]



    







