#ΪͼƬ��Ӹ�˹����������
import numpy as np
def add_gaussian_and_save(image):
    gaussian_noise_img = gaussian(image, 1)
    return gaussian_noise_img #������Ӹ�˹����֮���ͼƬ��

#������Ӹ�˹�����ĺ���,src�Ҷ�ͼƬ,scale������׼��
def gaussian(src, scale):
    gaussian_noise_img = np.copy(src) #���
    noise = np.random.normal(0, scale, size=(src.shape[1], src.shape[0],3)) #����
    add_noise_and_check = np.array(gaussian_noise_img, dtype=np.float32) #δ������ͼƬ
    add_noise_and_check += noise
    add_noise_and_check = add_noise_and_check.astype(np.int16)
    gaussian_noise_img = np.array(add_noise_and_check, dtype=np.uint8)
    return gaussian_noise_img