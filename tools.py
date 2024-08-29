from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import PIL
from PIL import Image
import torch
import cv2
import os
example_path = 'tyt.jpg'
example_image = Image.open(example_path)
def show_tensor(image, plot=False):
    '''把image对象或tensor对象变成图片'''
    if isinstance(image, torch.Tensor) or isinstance(image,np.ndarray): #or isinstance(image,PIL.JpegImagePlugin.JpegImageFile):
        im = image
        if isinstance(image,torch.Tensor) and image.requires_grad ==True:
            im = image.to('cpu').clone().detach()#从计算图中分离
        if not image.shape[-1]==3:
            trans = transforms.ToPILImage()#这里是把图像转换为channel后置的类型
            im = trans(image)
            plt.imshow(im)
            plt.axis('off')
            plt.show()
            return
        plt.imshow(im)
        plt.axis('off')
        plt.show()
    else:
        if plot == True:
            trans1 = transforms.ToTensor()
            trans2 = transforms.ToPILImage()
            im = trans2(trans1(image))
            plt.imshow(im)
            plt.axis('off')
            plt.show()
            return
        image.show()
        
train_data_process = transforms.Compose([
    #resnet网络输入的图像预处理
    transforms.Resize((256,256)),
    transforms.RandomRotation(45),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.1,saturation=0.1,hue=0.1),
    transforms.RandomGrayscale(p=0.025),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def toRGB(img):
    return img[:,:,::-1]

#@save

def compute_optical_flow(img1, img2):
    # 读取两张图片
    #img1 = cv2.resize(img1,(448,448))
    #img2 = cv2.resize(img2,(448,448))
    
    #cv2.destroyAllWindows()
   
   
    #cv2.destroyAllWindows()
    
    # 将图片转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 计算光流的幅度和角度
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 创建HSV图像来表示光流
    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255

    # 设置色调（H）为角度
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 设置亮度（V）为光流的幅度
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 将HSV图像转换为BGR图像以便显示
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr_flow

def video_processer(video_path, save_path,label,limit=1000):
    num_frame_scale = 5
    cap = cv2.VideoCapture(video_path)
    # get first video frame
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    i = 1
    last_frame=None
    while True:
        ok, frame = cap.read()
        if not ok:
            print('video is over')
            return
        l = len(str(i))
        assert l<=num_frame_scale#最多五位数的图片
        prefix = '0'*(num_frame_scale-l)
        cv2.imwrite(save_path+'/image/'+prefix+str(i)+'.jpg',frame)
        
        with open(save_path+'/label/'+prefix+str(i)+'.txt','w') as f:
            f.write(str(label))
            
        if i==1:
            last_frame  = frame.copy()
            i = i+1
            continue
        
        cv2.imwrite(save_path+'/optical_flow/'+prefix+str(i-1)+'.jpg',compute_optical_flow(frame,last_frame))
        i = i+1
        if i%100 ==0:
            print(f'img{i} processing')
        if (not limit==None) and i-1 == limit:
            print('reach the image limit')
            exit()
    