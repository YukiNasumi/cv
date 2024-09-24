import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
sys.path.append('../')
import tools
#backend_inline.set_matplotlib_formats('svg')
import argparse
from torchvision.models import ResNet18_Weights
import yaml


def train_model(epoch,model,device,trainloader,optimizer,creterion,save_path):
    model = model.to(device)
    for epoch in range(epoch):
        print('epoch {}:'.format(epoch+1))
        for i,(x, y) in enumerate(trainloader):
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = creterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1)%100==0:
                print(f'batch:{i+1}, loss:{loss.item()}')
        
    torch.save(model.state_dict(),save_path)
    #ax.plot(list(range(1,i+1+1)),losses)
    #plt.savefig('epoch{}'.format(epoch))
    
def test_model(model,loader,device):
    model.eval()
    model = model.to(device)
    tp,tn,fp,fn = 0,0,0,0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.cpu(), dim=1)
            labels = labels.cpu()
            tp += torch.sum((labels.reshape(-1) == 0) & (predicted == 0)).item()
            tn += torch.sum((labels.reshape(-1) == 1) & (predicted == 1)).item()
            fp += torch.sum((labels.reshape(-1) == 1) & (predicted == 0)).item()
            fn += torch.sum((labels.reshape(-1) == 0) & (predicted == 1)).item()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp) if ((tp+fp)!=0) else 0
    recall = tp/(tp+fn) if ((tp+fn)!=0) else 0
    print(f'accuracy: {100*accuracy}%') 
    print(f'precision: {100*precision}%')
    print(f'recall: {100*recall}%')
    print(f'F1-score: {(2*precision*recall/(precision+recall)) if ((precision+recall)!=0) else 0}')



if __name__ == '__main__':  
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path',type=str,required=False,help='保存模型的位置')
    parser.add_argument('--train_path',type=str,required=False,help='训练集的位置')
    parser.add_argument('--test_path',type=str,required=False,help='测试集的位置')
    parser.add_argument('--model_path',type=str,required=False,help='导入模型的路径')
    parser.add_argument('--config',type=str,required=False,help='配置文件的位置')
    
    args,unknown = parser.parse_known_args()
    

    from torchvision.models import resnet18
    model = tools.model_modify(resnet18(weights=ResNet18_Weights.DEFAULT),2)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

#hyper params
    args,unknown = parser.parse_known_args()
    save_path = args.save_path
    train_path = args.train_path
    test_path = args.test_path
    model_path = args.model_path
    config = args.config
    if train_path:
        if  config:
            with open(config,'r') as f:
                config = yaml.safe_load(f)
            lr = config.get('lr')
            epochs = config.get('epochs')
            optimizer = config.get('optimizer')
            criterion = config.get('criterion')
            device = config.get('device')
            if not device:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if 'adam' in optimizer.slower():
                optimizer=torch.optim.Adam(model.parameters(),lr)
            else:
                print("请检查配置文件的optimizer")
                exit
            if 'cross' in criterion.slower():
                criterion = nn.CrossEntropyLoss()
            else:
                print('请检查配置文件的criterion')
                exit
            batch_size = config.get('batch_size')
        else:
            print("请输入配置文件")
            exit
        trainloader = tools.get_loader(train_path,batch_size)
        train_model(epochs,model,device,trainloader,optimizer,criterion,save_path)

        if save_path:
            torch.save(model.state_dict(), save_path)
        
    if args.test_path:
        print('test on {}'.format(test_path))
        testloader = tools.get_loader(test_path,batch_size)
        test_model(model,testloader,device)