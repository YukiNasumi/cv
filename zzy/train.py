import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
sys.path.append('../')
import tools
from matplotlib_inline import backend_inline
#backend_inline.set_matplotlib_formats('svg')
import argparse



def train_model(epoch,model,device,trainloader,optimizer,creterion,save_path):
    model = model.to(device)
    for epoch in range(epoch):
        print('epoch {}:'.format(epoch))
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
    print(f'测试集准确率: {100*accuracy}%') 
    print(f'测试集精确率: {100*precision}%')
    print(f'测试集召回率: {100*recall}%')
    print(f'测试集F1值: {(2*precision*recall/(precision+recall)) if ((precision+recall)!=0) else 0}')



if __name__ == '__main__':  
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path',type=str,required=True,help='保存模型的位置')
    parser.add_argument('--train_path',type=str,required=False)
    parser.add_argument('--test_path',type=str,required=False)
    parser.add_argument('--model_path',type=str,required=False)
    parser.add_argument('--test',type=bool,required=True)
    parser.add_argument('--train',type=bool,required=True)

    args = parser.parse_args()
    test = args.test
    train = args.train
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}')


    from torchvision.models import resnet18
    model = tools.model_modify(resnet18(pretrained=True),2)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

#hyper params

    epochs = 2
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    creterion = nn.CrossEntropyLoss()
    batch_size = 10
    save_path = args.save_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}')

    if train:
        trainloader = tools.get_loader(args.train_path,batch_size)
        train_model(epochs,model,device,trainloader,optimizer,creterion,save_path)
    
        if save_path:
            torch.save(model, save_path)
        
    if test:
        testloader = tools.get_loader(args.test_path,batch_size)
        test_model(model,testloader,device)