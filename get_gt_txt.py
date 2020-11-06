#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET

classes = ["一次性快餐盒","书籍纸张","充电宝","剩饭剩菜","包","垃圾桶","塑料器皿","塑料玩具","塑料衣架","大骨头","干电池","快递纸袋","插头电线","旧衣服",
          "易拉罐","枕头","果皮果肉","毛绒玩具","污损塑料","污损用纸","洗护用品","烟蒂","牙签","玻璃器皿","砧板","筷子","纸盒纸箱","花盆","茶叶渣","菜帮菜叶",
          "蛋壳","调料瓶","软膏","过期药物","酒瓶","金属厨具","金属器皿","金属食品罐","锅","陶瓷器皿","鞋","食用油桶","饮料瓶","鱼骨"]
pth = r"F:/garbage_valid/"
# image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
image_ids = os.listdir("C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/trainval_garbage/garbage2020/Annotations/")
img_dir = os.listdir(pth)
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

with open("input/no2/no2.txt",'w') as f:
    for i in img_dir:
        flag = True
        for image_id in image_ids:
            if i.split(".")[0] == image_id.split(".")[0]:
                flag = False
                with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
                    root = ET.parse("C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/trainval_garbage/garbage2020/Annotations/"+image_id.split(".")[0]+".xml").getroot()
                    for obj in root.findall('object'):
                        if obj.find('difficult') != None:
                            difficult = obj.find('difficult').text
                            if int(difficult)==1:
                                continue
                        obj_name = obj.find('name').text
                        bndbox = obj.find('bndbox')
                        left = bndbox.find('xmin').text
                        top = bndbox.find('ymin').text
                        right = bndbox.find('xmax').text
                        bottom = bndbox.find('ymax').text
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                print(image_id)
                break
    if flag:
        f.write(i + '\n')


print("Conversion completed!")
