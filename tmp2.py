# # ----------------------------------------------------#
# #   获取测试集的ground-truth
# #   具体视频教程可查看
# #   https://www.bilibili.com/video/BV1zE411u7Vw
# # ----------------------------------------------------#
# import sys
# import os
# import glob
# import xml.etree.ElementTree as ET
#
# classes = [u"一次性快餐盒", u"书籍纸张", u"充电宝", u"剩饭剩菜", u"包", u"垃圾桶", u"塑料器皿", u"塑料玩具", u"塑料衣架", u"大骨头", u"干电池", u"快递纸袋",
#            u"插头电线", u"旧衣服",
#            u"易拉罐", u"枕头", u"果皮果肉", u"毛绒玩具", u"污损塑料", u"污损用纸", u"洗护用品", u"烟蒂", u"牙签", u"玻璃器皿", u"砧板", u"筷子", u"纸盒纸箱",
#            u"花盆", u"茶叶渣", u"菜帮菜叶",
#            u"蛋壳", u"调料瓶", u"软膏", u"过期药物", u"酒瓶", u"金属厨具", u"金属器皿", u"金属食品罐", u"锅", u"陶瓷器皿", u"鞋", u"食用油桶", u"饮料瓶",
#            u"鱼骨"]
#
# image_ids = os.listdir(r'F:/garbage_valid/')
#
# if not os.path.exists("./input"):
#     os.makedirs("./input")
# if not os.path.exists("./input/ground-truth"):
#     os.makedirs("./input/ground-truth")
#
# with open(r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/input/no2/no2.txt",'w') as f1:
#     for image_id in image_ids:
#         with open("./input/ground-truth/" + image_id + ".txt", "w") as new_f:
#
#             # if os.path(r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/trainval_garbage/garbage2020/Annotations/" + image_id + ".xml").exist():
#             # try:
#             root = ET.parser("trainval_garbage/garbage2020/Annotations/" + image_id.split(".")[0] + ".xml").getroot()
#             if not root:
#                 for obj in root.findall('object'):
#                     if obj.find('difficult') != None:
#                         difficult = obj.find('difficult').text
#                         if int(difficult) == 1:
#                             continue
#                     obj_name = obj.find('name').text
#                     bndbox = obj.find('bndbox')
#                     left = bndbox.find('xmin').text
#                     top = bndbox.find('ymin').text
#                     right = bndbox.find('xmax').text
#                     bottom = bndbox.find('ymax').text
#                     new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
#             else:
#                 f1.write(image_id.strip() + "\n")
#
#
# print("Conversion completed!")

#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#

# import sys
# import os
# import glob
# import xml.etree.ElementTree as ET
# import codecs
# classes = ["一次性快餐盒","书籍纸张","充电宝","剩饭剩菜","包","垃圾桶","塑料器皿","塑料玩具","塑料衣架","大骨头","干电池","快递纸袋","插头电线","旧衣服",
#           "易拉罐","枕头","果皮果肉","毛绒玩具","污损塑料","污损用纸","洗护用品","烟蒂","牙签","玻璃器皿","砧板","筷子","纸盒纸箱","花盆","茶叶渣","菜帮菜叶",
#           "蛋壳","调料瓶","软膏","过期药物","酒瓶","金属厨具","金属器皿","金属食品罐","锅","陶瓷器皿","鞋","食用油桶","饮料瓶","鱼骨"]
# pth = r"F:/garbage_valid/"
# # image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
# # image_ids = os.listdir("C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/trainval_garbage/garbage2020/Annotations/")
# p1 = "C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/trainval_garbage/garbage2020/Annotations/"
# img_dir = os.listdir(pth)
# if not os.path.exists("./input"):
#     os.makedirs("./input")
# if not os.path.exists("./input/ground-truth"):
#     os.makedirs("./input/ground-truth")
#
# #'20190816_095426'
# with open("input/no2/no2.txt",'w') as f:
#     for i in img_dir:
#         # flag = True
#         # for image_id in image_ids:
#         #     if i.split(".")[0] == image_id.split(".")[0]:
#         #         flag = False
#         temp_path = p1 + i.split(".")[0] + ".xml"
#         if not os.path.exists(temp_path):
#             f.write(temp_path + '\n')
#             continue
#         else:
#             with codecs.open("./input/ground-truth/"+i.split(".")[0]+".txt", "w","utf-8") as new_f:
#                 # root = ET.parse("C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/trainval_garbage/garbage2020/Annotations/"+image_id.split(".")[0]+".xml").getroot()
#                 root = ET.parse(temp_path).getroot()
#                 for obj in root.findall('object'):
#                     if obj.find('difficult') != None:
#                         difficult = obj.find('difficult').text
#                         if int(difficult)==1:
#                             continue
#                     obj_name = obj.find('name').text
#                     bndbox = obj.find('bndbox')
#                     left = bndbox.find('xmin').text
#                     top = bndbox.find('ymin').text
#                     right = bndbox.find('xmax').text
#                     bottom = bndbox.find('ymax').text
#                     new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
#             print(i)
#             # break
#
#         # if flag:
#
#
# print("Conversion completed!")

pth = r"F:/迅雷下载/cifar-10/data/cifar-10-batches-py/train_image_cifar10/"
new_pth = r"F:/迅雷下载/cifar-10/data/cifar-10-batches-py/one_class/"

import os
import matplotlib.pyplot as plt
from PIL import Image
d1 = os.listdir(pth)

for i in d1:
    if i.split(".")[1] == "txt":
        line = []
        with open(pth + i,'r') as f:
            line = f.readline()
            f.close()
        if line.split()[0] == '1':
            img = Image.open(pth + i.split(".")[0]+'.jpg')
            plt.imsave(new_pth + i.split(".")[0] + '.jpg',img)














