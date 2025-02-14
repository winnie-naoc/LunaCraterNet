import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
 
if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/11/yolo11.yaml')
    model.load('./yolo11l.pt') # loading pretrain weights
    model.train(data='./ultralytics/cfg/datasets/keng.yaml',
                cache=False,
                imgsz=1280,
                epochs=150,
                batch=8,
                close_mosaic=0,
                workers=11,
                # device='0',
                optimizer='SGD', # using SGD
                patience=50, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
