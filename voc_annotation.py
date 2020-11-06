# encoding:utf-8
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

from os import getcwd

classes = [u"一次性快餐盒", u"书籍纸张", u"充电宝", u"剩饭剩菜", u"包", u"垃圾桶", u"塑料器皿", u"塑料玩具", u"塑料衣架", u"大骨头", u"干电池", u"快递纸袋",
           u"插头电线", u"旧衣服",
           u"易拉罐", u"枕头", u"果皮果肉", u"毛绒玩具", u"污损塑料", u"污损用纸", u"洗护用品", u"烟蒂", u"牙签", u"玻璃器皿", u"砧板", u"筷子", u"纸盒纸箱",
           u"花盆", u"茶叶渣", u"菜帮菜叶",
           u"蛋壳", u"调料瓶", u"软膏", u"过期药物", u"酒瓶", u"金属厨具", u"金属器皿", u"金属食品罐", u"锅", u"陶瓷器皿", u"鞋", u"食用油桶", u"饮料瓶",
           u"鱼骨"]

sets1 = [('2020', 'trainval_garbage',)]

wd = getcwd()

def convert_annotation(year, image_id, list_file):
    # 要么在open打开的时候，用encoding='utf-8'
    # 要么直接用ET.parse直接读取xml文件，省了编码
    pth = '/home/ma-user/work/trainval_garbage/garbage%s/Annotations/%s.xml' % (year, image_id)
    infile = open(pth, 'r', encoding='utf-8')
    tree = ET.parse(infile)  # 直接用这个读取.xml文件，不要用 open 打开再读，就不会出现中文乱码
    root = tree.getroot()
    list_file.write('/home/ma-user/work/trainval_garbage/garbage%s/JPEGImages/%s.jpg' % (year, image_id))

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        # 检查该图片是否为voc里面的类别
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

for year, image_set in sets1:
    image_ids = open('/home/ma-user/work/trainval_garbage/garbage%s/ImageSets/Main/%s.txt' % (
    year, image_set)).read().strip().split()
    list_file = open('%s/%s_%s.txt' % (wd, year, image_set), 'w')

    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()

print("Finish!!!!!")
