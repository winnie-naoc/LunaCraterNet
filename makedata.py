import shutil
import os

# 配置目录路径
base_voc_dir = './train_data'
base_tmp_dir = './tmp'

file_list = ["train", "val", "test"]
for file in file_list:
    # 构建VOC/images和VOC/labels的完整路径
    images_dir = os.path.join(base_voc_dir, 'images', file)
    labels_dir = os.path.join(base_voc_dir, 'labels', file)
    
    # 如果目录不存在，则创建它们
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    # 构建tmp目录下文件的完整路径
    tmp_file_path = os.path.join(base_tmp_dir, f'{file}.txt')
    
    # 检查文件是否存在并读取
    if os.path.exists(tmp_file_path):
        print(os.path.exists(tmp_file_path))
        with open(tmp_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                print(line)
                line = line.rstrip('\n')
                # 复制图片到images目录
                shutil.copy(line, images_dir)
                # 替换文件扩展名，从jpg到txt，然后复制到labels目录
                label_path = line.replace('png', 'txt')
                shutil.copy(label_path, labels_dir)
    else:
        print(f'{tmp_file_path} does not exist.')