import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader
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

if __name__ == "__main__":
    dataset = torchvision.datasets.ImageFolder(root='data/',transform=train_data_process)

    
    dloader = DataLoader(dataset,batch_size=10,shuffle=True, num_workers=4)

