import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader,random_split
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_valid_split(data_set, valid_ratio):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) #compute the size of validation set
    train_set_size = len(data_set) - valid_set_size #compute the size of training set
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(0))
    return np.array(train_set),np.array(valid_set)#!!!convert to np.array(or tensor)!!!

class CovidDataset(Dataset):
    def __init__(self,sample,label):
        if label is None:#!!!use 'is' rather than 'equal to' !!!
            self.label=None
        else:
            self.label=torch.FloatTensor(label)
        self.sample=torch.FloatTensor(sample)#!!!convert to float!!!
    def __getitem__(self, index):
        if self.label is None:
            return self.sample[index]
        else:
            return self.sample[index],self.label[index]
    def __len__(self):
        return len(self.sample)

class CovidPredModel(nn.Module):
    def __init__(self,input_dim):
        super(CovidPredModel,self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,sample):
        prediction=self.layers(sample)
        prediction=prediction.squeeze(1)
        return prediction

def trainer(train_loader,valid_loader,model,n_epoches,learning_rate,model_path):
    writer=SummaryWriter()
    pred_train=[]
    pred_valid=[]
    loss_train=[]
    loss_valid=[]
    criterion=nn.MSELoss(reduction='mean')#!!!if reduction is 'none', then loss will be a vector!!!
    optimizer=optim.SGD(model.parameters(),lr=learning_rate)
    count=0
    for epoch in range(n_epoches):#!!!the data type of n_epoches is int, which is not iteratable!!!
        model.train()
        pbar_train=tqdm(train_loader,leave=True)
        for sample,label in pbar_train:#!!!one batch per iteration!!!
            optimizer.zero_grad()
            prediction=model(sample)
            loss=criterion(prediction,label)
            loss.backward()
            optimizer.step()#update wights and biases
            count+=1#!!!the amount of iteration!!!
            loss_train.append(loss.detach().numpy())
            pred_train.append(prediction.detach().numpy())#!!!first detaching from computegraph, then convert to numpy array!!!
            pbar_train.set_description(f'Epoch:{epoch+1}/{n_epoches}')
            pbar_train.set_postfix(Loss=loss.detach().item())
        mean_loss_train=sum(loss_train)/len(loss_train)#!!!the average loss in each epoch!!!
        writer.add_scalar('train loss',mean_loss_train,count)#!!!add average loss and the amount of samples of each epoch to TRAIN LOSS file!!! 
        model.eval()
        for sample,label in valid_loader:
            with torch.no_grad():
                prediction=model(sample)
                loss=criterion(prediction,label)
            pred_valid.append(prediction.detach().numpy())#!!!append function for list returns None, whlie append function for numpy returns value
            loss_valid.append(loss.detach().item())
        mean_loss_valid=sum(loss_valid)/len(loss_valid)
        writer.add_scalar('validation loss',mean_loss_valid,count)
    torch.save(model.state_dict(),model_path)#save trained model
    return pred_train,pred_valid

def predictor(test_loader,model):
    pred_test=[]
    model.eval()
    count=0
    pbar_test=tqdm(test_loader,leave=True)
    for sample in pbar_test:
        with torch.no_grad():
            prediction=model(sample)
            pred_test.append(prediction.detach().numpy())
            count+=1
        pbar_test.set_description(f'iteration {count}:')
    return pred_test

config={'path':"D:\\study\\大四上\\ML\\HW1\\data\\",
    'file_train':'covid.train_new.csv',
    'file_test':'covid.test_un.csv',
    'n_epoches':3000,
    'batch_size':256,
    'learning_rate':1e-6,
    'valid_ratio':0.25,
    'model_path':"D:\\study\\大四上\\ML\\HW1\\model\\covid_model.pt",
    'result_path':"D:\\study\\大四上\\ML\\HW1\\result\\result.csv"}

if __name__=="__main__":
    data_train,data_test=pd.read_csv(os.path.join(config['path'],config['file_train'])).values,pd.read_csv(os.path.join(config['path'],config['file_test'])).values
    train_set,valid_set=train_valid_split(data_train,config['valid_ratio'])
    sample_train,label_train=train_set[:,:-1],train_set[:,-1]
    sample_valid,label_valid=valid_set[:,:-1],valid_set[:,-1]
    sample_test,label_test=data_test,None
    dataset_train,dataset_valid,dataset_test=CovidDataset(sample_train,label_train),CovidDataset(sample_valid,label_valid),CovidDataset(sample_test,label_test)
    loader_train=DataLoader(dataset_train,batch_size=128,shuffle=True)
    loader_valid=DataLoader(dataset_valid,batch_size=128,shuffle=True)
    loader_test=DataLoader(dataset_test,batch_size=128,shuffle=True)
    covid_model=CovidPredModel(input_dim=sample_train.shape[1])
    pred_train,pred_valid=trainer(loader_train,loader_valid,covid_model,config['n_epoches'],config['learning_rate'],config['model_path'])
    print('training finished!')

    state_dict_test=torch.load(config['model_path'])
    covid_model=CovidPredModel(input_dim=sample_train.shape[1])
    covid_model.load_state_dict(state_dict_test)
    pred_test=predictor(loader_test,covid_model)
    pred_test=pd.Series(pred_test).to_csv(config['result_path'])
    print('test finished!')