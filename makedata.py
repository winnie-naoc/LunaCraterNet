import shutil
import os
 
file_List = ["train", "val", "test"]
for file in file_List:
    if not os.path.exists('../VOC/images/%s' % file):
        os.makedirs('../VOC/images/%s' % file)
    if not os.path.exists('../VOC/labels/%s' % file):
        os.makedirs('../VOC/labels/%s' % file)
    print(os.path.exists('../tmp/%s.txt' % file))
    f = open('../tmp/%s.txt' % file, 'r')
    lines = f.readlines()
    for line in lines:
        print(line)
        line = line.rstrip('\n') 
        #line = "/".join(line.split('/')[-5:]).strip()
        shutil.copy(line, "../VOC/images/%s" % file)
        line = line.replace('jpg', 'txt')
        shutil.copy(line, "../VOC/labels/%s/" % file)