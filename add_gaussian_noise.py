#为图片添加高斯噪声并保存
import numpy as np
def add_gaussian_and_save(image):
    gaussian_noise_img = gaussian(image, 1)
    return gaussian_noise_img #返回添加高斯噪声之后的图片集

#定义添加高斯噪声的函数,src灰度图片,scale噪声标准差
def gaussian(src, scale):
    gaussian_noise_img = np.copy(src) #深拷贝
    noise = np.random.normal(0, scale, size=(src.shape[1], src.shape[0],3)) #噪声
    add_noise_and_check = np.array(gaussian_noise_img, dtype=np.float32) #未经检查的图片
    add_noise_and_check += noise
    add_noise_and_check = add_noise_and_check.astype(np.int16)
    gaussian_noise_img = np.array(add_noise_and_check, dtype=np.uint8)
    return gaussian_noise_img