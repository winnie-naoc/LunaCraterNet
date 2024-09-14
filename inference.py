'''
启动命令：
python your_script.py --images-dir="/path/to/images" --target-dir="/path/to/target" --onnx-model-path="/path/to/model.onnx" --conf-thre=0.25
'''
import os
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '2000000000')
import cv2

import numpy as np
import onnxruntime as rt
import argparse
import json
import base64
import copy

labelme_content = {
  "version": "4.5.6",
  "flags": {},
  "shapes":[],
  "imagePath": "xx.jpg",
  "imageData":"", 
  "imageHeight": 2736,
  "imageWidth": 3648
}


CLASSES = {
    0: 'keng'
}
def cal_iou(box1,box2):
    i_xmin = max(box1[0], box2[0])
    i_ymin = max(box1[1], box2[1])
    i_xmax = min(box1[2], box2[2])
    i_ymax = min(box1[3], box2[3])
    inter_w = max(i_xmax - i_xmin , 0)
    inter_h = max(i_ymax - i_ymin , 0)
 
    i = inter_w * inter_h
    if(0 == i):
        return 0
    else:
        u = (box1[2] - box1[0] ) * (box1[3] - box1[1] ) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - i
        iou = i / u
        return iou

def cal_iou_2(box1,box2):
    i_xmin = max(box1[0], box2[0])
    i_ymin = max(box1[1], box2[1])
    i_xmax = min(box1[2], box2[2])
    i_ymax = min(box1[3], box2[3])
    inter_w = max(i_xmax - i_xmin , 0)
    inter_h = max(i_ymax - i_ymin , 0)
    area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    inter_a = inter_w * inter_h
    if(0 == area_1*area_2):
        return 0
    if(inter_a/area_1 > 0.5 or inter_a/area_2 > 0.5):
        return 1
    else:
        return 0

        
def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
 
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
 
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
 
def onnx_inf(onnxModulePath, data):
    sess = rt.InferenceSession(onnxModulePath)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
 
    pred_onnx = sess.run([output_name], {input_name: data.reshape(1, 3, 1280, 1280).astype(np.float32)})
 
    return pred_onnx
 
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # isinstance 用来判断某个变量是否属于某种类型
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
 
def nms_boxes(boxes, scores):
 
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
 
    areas = w * h
    order = scores.argsort()[::-1]
 
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
 
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
 
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.45)[0]
 
        order = order[inds + 1]
    keep = np.array(keep)
    return keep
 
def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
 
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
 
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
 
    # Settings
    max_wh = 20  # (pixels) maximum box width and height
    max_nms = 1000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
 
    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm))] * bs
 
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros(len(lb), nc + nm + 5)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)
 
        # If none remain process next image
        if not x.shape[0]:
            continue
 
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
 
        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks
 
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
 
        else:  # best class only
            conf = np.max(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            j = np.argmax(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(box.shape[:1][0]) > conf_thres]
 
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes, device=x.device)).any(1)]
 
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        index = x[:, 4].argsort(axis=0)[:max_nms][::-1]
        x = x[index]
 
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_boxes(boxes, scores)
        i = i[:max_det]  # limit detections
 
        # 用来合并框的
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
 
        output[xi] = x[i]
 
    return output
 
def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
 
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
 
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
 
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
 

class MoonDedectionModel(object):
    def __init__(self,model_path) -> None:
        print("MoonDedectionModel loading ....")
        self.names =  ['keng', 'other']
        self.img_size = (1280,1280)
        onnxModulePath = model_path#"/DATA01/yolov5-7.0-train/opt_models/gray_zoon_0521/best.onnx"
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = session = rt.InferenceSession(onnxModulePath, providers=providers)
        self.output_names = [x.name for x in session.get_outputs()]
        self.input_name = session.get_inputs()[0].name

    def preprocess(self,img):
        img = cv2.resize(img, self.img_size)
    
        # preprocess
        im = letterbox(img, self.img_size, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype(np.float32)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def inference(self,img):
        '''
        input:
              img: local path  or np.ndarray of image
        output:
        '''
        im = self.preprocess(img)
        pred = self.session.run(self.output_names, {self.input_name: im.reshape(1, 3, 1280, 1280).astype(np.float32)})
        
        conf_thres = 0.1  # confidence threshold
        iou_thres = 0.5  # NMS IOU threshold
        max_det = 2000  # maximum detections per image
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
        # Process predictions
        seen = 0
        for i, det in enumerate(pred):  # per image
            seen += 1
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
        # print(pred)
        outputs = pred[0][:, :6]
        ret=[]
        count_list = [0,0]
        if len(outputs[:, 4:] > 0):
            for i in outputs:
                prob = i[4]
                cls = int(i[5])
                prob = np.around(prob, decimals=2)
                if prob >= conf_thres:
                    all_pred_boxes = i[:4]
                    # x1 = int(all_pred_boxes[0])
                    # y1 = int(all_pred_boxes[1])
                    # x2 = int(all_pred_boxes[2])
                    # y2 = int(all_pred_boxes[3])
                    x1 = max(0,int(all_pred_boxes[0]))
                    y1 = max(0,int(all_pred_boxes[1]))
                    x2 = min(int(all_pred_boxes[2]),1280)
                    y2 = min(int(all_pred_boxes[3]),1280)
                    if (x2-x1)*7>400:
                        count_list[0] += 1
                    else:
                        count_list[1] += 1

                    box_ret = {"bbox":[x1,y1,x2,y2],"prob":prob}
                    ret.append(box_ret)
                        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        # cv2.putText(img, CLASSES[cls]+' '+str(prob), (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 1, 4)
                        # cv2.imwrite('/data1/dengtao/develop_dt/yolov5-train/yolov5-7.0/opt_models/ret.jpg', img)
        print("count_list:",count_list)
        return ret,count_list
    
def draw_bbox(image,box,prob):

    bbox_w = (box[2] - box[0])*7
    bbox_h = (box[3] - box[1])*7
    if bbox_h == 0 or bbox_w ==0:
        return
    ratio = bbox_w/bbox_h
    if ratio > 1.3 or ratio<0.75:
        return
    
    cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
    if bbox_w > 400:
        cv2.putText(image, str(bbox_w), (box[0],box[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 1, 4)
    else:
        cv2.putText(image, str(bbox_w), (box[0],box[1]), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 1, 4)

def is_valid_box(box):
    bbox_w = (box[2] - box[0])*7
    bbox_h = (box[3] - box[1])*7
    if bbox_h < 2 or bbox_w <2:
        return False
    ratio = bbox_w/bbox_h
    if ratio > 1.3 or ratio<0.75:
        return False
    return True
    
def box_duplicate_removal(bbox_dp,row,col):
    index = str(row) + '_' + str(col)
    last_indexs = [
        str(row) + '_' + str(col-1),
        str(row-1) + '_' + str(col),
        str(row-1) + '_' + str(col-1),
        str(row-1) + '_' + str(col+1),
    ]
    for idx in last_indexs:  # 遍历相邻的个区域
        if idx in bbox_dp: # 如果该区域存在
            for tbox in bbox_dp[idx]:
                for box in bbox_dp[index]:  # 遍历当前区域内的所有box
                    if cal_iou_2(tbox['bbox'], box['bbox']) > 0.1:
                        if tbox['prob'] > box['prob']:
                            bbox_dp[index].remove(box)
                        else:
                            if tbox in bbox_dp[idx]:
                                bbox_dp[idx].remove(tbox)




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
def load_image_base64(img_path: str):
    """
    Loads an encoded image as an array of bytes.
    
    """
    with open(img_path, "rb") as f:
        #img = base64.b64encode(f.read())
        img = base64.b64encode(f.read()).decode('utf-8')
        return img
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some paths.')  
    parser.add_argument('--images-dir', type=str, help='Path to the images directory.')  
    parser.add_argument('--target-dir', type=str, help='Path to the target directory.')  
    parser.add_argument('--onnx-model-path', type=str, help='Path to the ONNX model.')  
    parser.add_argument('--conf-thre', type=float, help='Confidence threshold.')  
      
    args = parser.parse_args()  
      
    images_dir = args.images_dir if args.images_dir else "/DATA02/CE2_7m_JPG/I区/"  
    target_dir = args.target_dir if args.target_dir else "/DATA02/CE2_7m_JPG/results/I区/"  
    onnx_model_path = args.onnx_model_path if args.onnx_model_path else "/DATA01/yolov5-7.0-train/train_records/0629/weights/best.onnx"  
    conf_thre = args.conf_thre if args.conf_thre else 0.15  

    pdm = MoonDedectionModel(onnx_model_path)

    gray_files = os.listdir(images_dir)
    backimg_arr = []
    cnt = 0
    for grayimg_file in gray_files:
        # if cnt >1:
        #     break
        cnt += 1
        img_file = os.path.join(images_dir,grayimg_file)
        try:
            ori_img = cv2.imread(img_file)
        except Exception as e:
            print("error:",str(e))
            continue
        img_h,img_w= ori_img.shape[:-1]
        win_size = 1280
        overlap_size = 280
        step = 1000
        iter_rows = (img_h + step - 1)//step
        iter_cols = (img_w + step - 1)//step
        print("iter_rows*iter_cols:{}*{}={} ",iter_rows,iter_cols,iter_rows*iter_cols)
        bbox_dp={}
        count_list = [0,0]
        for row in range(iter_rows):
            for col in range(iter_cols):
                offset_y = row*step
                offset_x = col*step
                img = ori_img[row*step:row*step+win_size,col*step:col*step + win_size]
                bboxs,cnt_list = pdm.inference(img)
                #bbox_dp[str(row)+'_'+str(col)] = bboxs
                r_bboxes = []
                for item in bboxs:
                    if item['prob'] < conf_thre:
                        del item
                        continue
                    t_box = item['bbox']
                    item['bbox'] = [offset_x + t_box[0],offset_y + t_box[1],offset_x + t_box[2],offset_y + t_box[3]]
                    #draw_bbox(img,item["bbox"],item["prob"])
                    r_bboxes.append(item)
                print("file: {}, row : {}, col: {}".format(grayimg_file,row,col))
                count_list[0] += cnt_list[0]
                count_list[1] += cnt_list[1]

                
                bbox_dp[str(row)+'_'+str(col)] = r_bboxes
                box_duplicate_removal(bbox_dp,row,col)



        for idx in bbox_dp:
            for item in bbox_dp[idx]:
                draw_bbox(ori_img,item["bbox"],item["prob"])


        target_file = os.path.join(target_dir,"result_" + grayimg_file)

        # 生成与图像对应的标注文件
        new_data = copy.deepcopy(labelme_content)
        tt_shapes = []
        for idx in bbox_dp:
            for item in bbox_dp[idx]:
                if not is_valid_box(item["bbox"]):
                    continue
                ttbox = generate_one_label("keng",item["bbox"])
                tt_shapes.append(ttbox)
        new_data["shapes"] = tt_shapes
        new_data["imagePath"]= img_file
        new_data["imageHeight"] = img_h
        new_data["imageWidth"] = img_w
        new_data["imageData"] = load_image_base64(img_file)
        target_json_file = target_file.replace(".jpg",".json")
        label_json_str = json.dumps(new_data)
        with open(target_json_file,'w') as f:
            f.write(label_json_str)
