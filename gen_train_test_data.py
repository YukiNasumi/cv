import os
import shutil
import random
from torch.utils.data import DataLoader
import torchvision
import gen_dataloader
def random_select_files(source_dir, target_dir, num_files):
    # 获取源目录中的所有文件
    all_files = os.listdir(source_dir)

    # 从中随机选择若干文件
    selected_files = random.sample(all_files, num_files)

    # 将选定的文件复制到目标目录
    for file_name in selected_files:
        full_file_name = os.path.join(source_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.move(full_file_name, target_dir)
            


'''os.mkdir('./data_train/')
os.makedirs('./data_test/concert')
os.makedirs('./data_test/non-concert')
random_select_files('./data/concert/','./data_test/concert',1000)
random_select_files('./data/non-concert/','./data_test/non-concert',100)
shutil.copytree('./data/concert','./data_train/concert')
shutil.copytree('./data/non-concert', './data_train/non-concert')'''
dataset_train = torchvision.datasets.ImageFolder(root='data_train/',transform=gen_dataloader.train_data_process)
train_loader = DataLoader(dataset_train,batch_size=10,shuffle=True, num_workers=4)
dataset_test = torchvision.datasets.ImageFolder(root='data_test/',transform=gen_dataloader.train_data_process)
train_loader = DataLoader(dataset_test,batch_size=10,shuffle=True, num_workers=4)
    