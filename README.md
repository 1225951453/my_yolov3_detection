# my_yolov3_detection

##所需环境
pytorcch==1.4.0

###预测步骤
采用yolov3进行检测，使用Adam优化器，余弦退火学习率衰减策略以及热重启。kaiming_normal初始化方式初始化模型参数
【1】训练时加入实时数据增强。

【2】添加高斯噪声，模型预测效果下降14.56%。
