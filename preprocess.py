import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 256


# 按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
 
    # 获取图像尺寸
    h, w = image.shape[:2]
 
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)
 
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
 
    # RGB颜色
    black = [0, 0, 0]
 
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=black)
 
    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))
    
 
i = 1
for dir_item in os.listdir('./pics/xianluo/train/'):
    # 从初始路径开始叠加，合并成可识别的操作路径
    full_path = './pics/xianluo/train/'+dir_item
 
    #if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
        #read_path(full_path)
         #else:  # 文件
    #if dir_item.endswith('.JPEG'):
    #print(full_path)
    image = cv2.imread(full_path)
    #cv2.imshow('img', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
    cv2.imwrite('./pics/xianluo/trainAfter/'+str(i)+'.jpg', image)
    i+=1
    #images.append(image)
    #labels.append(path_name)
 
 
'''
 # 从指定路径读取训练数据
def load_dataset(path_name):
    images = read_path(path_name)
 
     # 将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
     # 我和闺女两个人共1200张图片，IMAGE_SIZE为64，故对我来说尺寸为1200 * 64 * 64 * 3
     # 图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    print(images.shape)
 
     # 标注数据
    #labels = np.array([0 if label.endswith('me') else 1 for label in labels])
 
    return images #labels
'''

'''
if __name__ == '__main__':
    #img =  cv2.imread(path+'000891.jpg')
    #image = resize_image(img)
    #cv2.imshow('img', image)
    cv2.imwrite("D:\\1.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
     if len(sys.argv) != 2:
         print("Usage:%s path_name\r\n" % path)
     else:
         images = load_dataset(path)
'''