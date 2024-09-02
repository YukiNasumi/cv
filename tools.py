from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import PIL
from PIL import Image
import torch
import cv2
import os
import random
import shutil
import torchvision
import copy
from torch.utils.data import Dataset,DataLoader
def show_tensor(image, plot=True):
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
        plt.imshow(im)#imshow要求channel后置
        plt.axis('off')
        plt.show()
    else:#是通过PIL.Image.open(path)打开的图像,自动即为channel后置
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
    transforms.Resize((448,448)),
    transforms.RandomRotation(45),
    #transforms.CenterCrop(448),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.1,saturation=0.1,hue=0.1),
    transforms.RandomGrayscale(p=0.025),
    transforms.ToTensor(),
    #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
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

def video_processer(video_path,save_path,label,limit=None,init_index=1,num_frame_scale=5):
    """_summary_
    在save_path下生成三个目录（如果没有就自动创建）：image,label,optical_flow。
    分别用来存放图片，标签，和光流处理后的图像
    Args:
        video_path (_str_): 
        save_path (_str_): 
        label (_int_): 
        limit (int, optional): 限制生成多少副原图，默认None不限制 
        init_index (int, optional): 图像的编号，光流图会比原图少1
        num_frame_scale:图像编号长度
    """
    prefix = lambda i:'0'*(num_frame_scale-len(str(i)))+str(i)
    cap = cv2.VideoCapture(video_path)
    # get first video frame
    if not cap.isOpened():
        raise FileNotFoundError("Error: Could not open video.")
    i = init_index
    last_frame=None
    if not (os.path.exists(save_path) and os.path.isdir(save_path)):
        os.makedirs(save_path)
    for dir in ('image','label','optical_flow'):
        if not os.path.exists(os.path.join(save_path,dir)):
            os.mkdir(os.path.join(save_path,dir))
    while True:
        ok, frame = cap.read()
        if not ok:
            print('video is over,the last index is {}'.format(i-1))
            return i-1 #返回最后一个编号
        l = len(str(i))
        assert l<=num_frame_scale#最多五位数的图片
        cv2.imwrite(save_path+'/image/'+prefix(i)+'.jpg',frame)
        with open(save_path+'/label/'+prefix(i)+'.txt','w') as f:
            f.write(str(label))
            
        if i==init_index:
            last_frame  = frame.copy()
            i = i+1
            continue
        
        cv2.imwrite(save_path+'/optical_flow/'+prefix(i-1)+'.jpg',compute_optical_flow(frame,last_frame))
        i = i+1
        if i%100 ==0:
            print(f'img{i} processing')
        if (not limit==None) and i-1 == limit:
            print('reach the image limit')
            exit()
        
    
def gen_dataloader(source_path1,source_path2,scale=1000,data_path='.',batch_size=10,shuffle=True,data_train = '__train',data_test = '__test'):
    """_summary_
    在两个source_path下的optical_flow路径下分别随机采集数量为scale的图像
    在根目录下创建data目录，依托data产生训练集和测试集以及相应的dataloader
    Args:
        source_path1 (_type_): 直接输入有图片的目录
        source_path2 (_type_): 同path1
        scale (int, optional): _description_. Defaults to 1000.
        data_path:存放数据的目录
        batch_size (int, optional): _description_. Defaults to 10.
        shuffle (bool, optional): _description_. Defaults to True.
        data_train (str, optional): _description_. Defaults to '__train'.
        data_test (str, optional): _description_. Defaults to '__test'.
    """
    #首先组织目录
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    data_train = os.path.join(data_path,data_train)
    data_test = os.path.join(data_path,data_test)
    os.mkdir(data_train)
    os.mkdir(data_test)

    #检查源路径是否合法
    target1 = source_path1
    target2 = source_path2
    if not (os.path.exists(target1) and os.path.exists(target2)):
        raise FileNotFoundError('illegal input dir')
    
    #选择scale规模的数据放入data_train中
    train_data_file_names=[]#记录放入训练集的文件，后面防止将这些文件又放入测试集
    for i,dir in enumerate((target1,target2)):
        all_file = os.listdir(dir)
        train_data_file_name = random.sample(all_file,scale)
        train_data_file_names.append(train_data_file_name)
        os.mkdir(os.path.join(data_train,str(i)))
        for file in train_data_file_name:
            full_file_name = os.path.join(dir, file)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name,os.path.join(data_train,str(i)))
    dataset_train = torchvision.datasets.ImageFolder(root=data_train,transform=train_data_process)
    train_loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=shuffle, num_workers=4)
    
    #接下来构建测试集
    for i,(dir,train_names) in enumerate(zip((target1,target2),train_data_file_names)):
        all_file = [d for d in os.listdir(dir) if not d in train_names]
        test_data_file_name = random.sample(all_file,scale//10)#测试集规模是训练集的十分之一
        os.mkdir(os.path.join(data_test,str(i)))
        for file in test_data_file_name:
            full_file_name = os.path.join(dir, file)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name,os.path.join(data_test,str(i)))
    dataset_test = torchvision.datasets.ImageFolder(root=data_test,transform=train_data_process)
    test_loader = DataLoader(dataset_test,batch_size=batch_size,shuffle=shuffle, num_workers=4)
    return (dataset_train,train_loader),(dataset_test,test_loader)

class data_pro:
    def __init__(self,last_idx=0):
        self.last_idx=last_idx#原图最新一张的索引,默认为0，则从1开始计数
        pass
    
    def video_process(self,video_path,save_path,label,limit=None,init_index=None,num_frame_scale=5):
        """_summary_

        Args:
            video_path (_type_): _description_
            save_path (_type_): _description_
            label (_type_): _description_
            limit (_type_, optional): _description_. Defaults to None.
            init_index (_type_, optional): _description_. 默认自动从上一次生成的开始编号，如果切换save_path则需要手动设置编号，以防覆盖
            num_frame_scale (int, optional): _description_. Defaults to 5.
        """
        if not init_index:
            init_index = self.last_idx+1
        self.last_idx=video_processer(video_path,save_path,label,limit=None,init_index=init_index,num_frame_scale=5)
    
    def data_generate(self,source_path1,source_path2,scale=1000,data_path='.',batch_size=10,shuffle=True,data_train = '__train',data_test = '__test'):
        """_summary_
    在两个source_path下的optical_flow路径下分别随机采集数量为scale的图像
    在根目录下创建data目录，依托data产生训练集和测试集以及相应的dataloader
    Args:
        source_path1 (_type_): _description_
        source_path2 (_type_): _description_
        scale (int, optional): _description_. Defaults to 1000.
        data_path:存放数据的目录
        batch_size (int, optional): _description_. Defaults to 10.
        shuffle (bool, optional): _description_. Defaults to True.
        data_train (str, optional): _description_. Defaults to 'data_train'.
        data_test (str, optional): _description_. Defaults to 'data_test'.
    """
        (train,test)=gen_dataloader(source_path1,source_path2,scale=1000,data_path='.',batch_size=10,shuffle=True,data_train = '__train',data_test = '__test')
        self.dataset_train,self.train_loader = train
        self.dataset_test,self.test_loader = test
        self.classes = self.dataset_test.classes
        self.class2idx = self.dataset_test.class_to_idx

def model_modify(amodel,class_num):
    """_summary_
        将模型的最后一层改变为要求的分类数

    Args:
        amodel (_type_): _description_
        class_num (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = copy.deepcopy(amodel)
    model.fc = torch.nn.Linear(model.fc.in_features, class_num)
    return model
    
def get_loader(src,batch_size,transform=train_data_process):
    dataset_test = torchvision.datasets.ImageFolder(root=src,transform=transform)
    return DataLoader(dataset_test,batch_size=batch_size,shuffle=True, num_workers=4)

def move_sample(src,dst,n_sample=None):
    """把src中数量为sample的放入dst中

    Args:
        src (_type_): _description_
        dst (_type_): _description_
        n_sample (_type_): _description_
    """
    all_name = os.listdir(src)
    if not n_sample:
        selected = all_name
    else :
        selected = random.sample(all_name,n_sample)
    for n in selected:
        full_name = os.path.join(src,n)
        shutil.copy(full_name,dst)
        
        
def sample_assgin(src,dst1,dst2,rate):
    """_summary_

    Args:
        src (_type_): _description_
        dst1 (_type_): _description_
        dst2 (_type_): _description_
        rate (_type_): _description_
    """
    assert rate<=1 and rate>=0
    all_file = os.listdir(src)
    random.shuffle(all_file)
    l = len(all_file)
    for i in range(int(l*rate)):
        full_name = os.path.join(src,all_file[i])
        shutil.copy(full_name,dst1)
    for i in range(int(l*rate),l):
        full_name = os.path.join(src,all_file[i])
        shutil.copy(full_name,dst2)