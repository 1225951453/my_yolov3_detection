import os
import random 
random.seed(0)

xmlfilepath=r'./VOCdevkit/VOC2007_trainval/Annotations'
saveBasePath=r"./VOCdevkit/VOC2007_trainval/ImageSets/new_main/"
label_path = r"./VOCdevkit/VOC2007_trainval/ImageSets/Layout/"
label_path_test = r"./VOCdevkit/VOC2007_test/ImageSets/Layout/"
trainval_percent=1
train_percent=1

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

# num=len(total_xml)  
# list=range(num)  
# tv=int(num*trainval_percent)  
# tr=int(tv*train_percent)  
# trainval = random.sample(list,tv)  
# train = random.sample(trainval,tr)  
label_train = open(os.path.join(label_path,"train.txt"),'r')
label_trainval = open(os.path.join(label_path,"trainval.txt"),'r')
label_val = open(os.path.join(label_path,"val.txt"),"r")
label_test = open(os.path.join(label_path_test,"test.txt"),"r")

print("train and val size",tv)
print("train size",tr)
# ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
# ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
# ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
# fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  #这么做是为了排除不合规的数据，有些输入不在5011之内
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        #下面的 else 有意义吗?
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
