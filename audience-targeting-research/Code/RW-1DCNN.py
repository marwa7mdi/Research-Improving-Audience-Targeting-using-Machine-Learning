# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:43.847156Z","iopub.execute_input":"2024-02-18T09:40:43.847518Z","iopub.status.idle":"2024-02-18T09:40:43.854920Z","shell.execute_reply.started":"2024-02-18T09:40:43.847490Z","shell.execute_reply":"2024-02-18T09:40:43.853476Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
    
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:43.856621Z","iopub.execute_input":"2024-02-18T09:40:43.857020Z","iopub.status.idle":"2024-02-18T09:40:45.538603Z","shell.execute_reply.started":"2024-02-18T09:40:43.856991Z","shell.execute_reply":"2024-02-18T09:40:45.537155Z"}}
!conda install -y gdown

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:45.540724Z","iopub.execute_input":"2024-02-18T09:40:45.541150Z","iopub.status.idle":"2024-02-18T09:40:45.847031Z","shell.execute_reply.started":"2024-02-18T09:40:45.541109Z","shell.execute_reply":"2024-02-18T09:40:45.845921Z"}}
!gdown --id 1uzho8MutOmyjeGIf4wezbuIwt5R204ft

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:45.849385Z","iopub.execute_input":"2024-02-18T09:40:45.849712Z","iopub.status.idle":"2024-02-18T09:40:46.156185Z","shell.execute_reply.started":"2024-02-18T09:40:45.849660Z","shell.execute_reply":"2024-02-18T09:40:46.155372Z"}}
!gdown --id 1-HWTeDoeOhthPZY0roptxCaXaiBetIDW

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:46.157479Z","iopub.execute_input":"2024-02-18T09:40:46.157767Z","iopub.status.idle":"2024-02-18T09:40:46.470549Z","shell.execute_reply.started":"2024-02-18T09:40:46.157740Z","shell.execute_reply":"2024-02-18T09:40:46.467233Z"}}
!gdown --id 1FaF5FAx9ztq4Qmxe52gBwVKFYBWQ56nD

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:46.474373Z","iopub.execute_input":"2024-02-18T09:40:46.474988Z","iopub.status.idle":"2024-02-18T09:40:46.481283Z","shell.execute_reply.started":"2024-02-18T09:40:46.474927Z","shell.execute_reply":"2024-02-18T09:40:46.480478Z"}}
model='/kaggle/input/new-aitm-model/new_AITM.model'
model_file = '/kaggle/working/new_AITM.model'
val_path='/kaggle/working/ctr_cvr.dev'
test_path='/kaggle/working/ctr_cvr.test'
train_path='/kaggle/working/ctr_cvr.train'
sample_50='/kaggle/input/sample-50/sample_50.csv'


# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:46.482543Z","iopub.execute_input":"2024-02-18T09:40:46.482819Z","iopub.status.idle":"2024-02-18T09:40:46.495564Z","shell.execute_reply.started":"2024-02-18T09:40:46.482796Z","shell.execute_reply":"2024-02-18T09:40:46.494278Z"}}
half_of_data='19,035,335'

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:46.498017Z","iopub.execute_input":"2024-02-18T09:40:46.498450Z","iopub.status.idle":"2024-02-18T09:40:50.330508Z","shell.execute_reply.started":"2024-02-18T09:40:46.498413Z","shell.execute_reply":"2024-02-18T09:40:50.327728Z"}}
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import re
import random
# data loader to load the data to the model
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# train
import sys
from sklearn.metrics import roc_auc_score


from sklearn.metrics import roc_curve


# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.331473Z","iopub.status.idle":"2024-02-18T09:40:50.331890Z","shell.execute_reply.started":"2024-02-18T09:40:50.331700Z","shell.execute_reply":"2024-02-18T09:40:50.331724Z"}}
use_columns = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301']
vocabulary_size = {
    '101': 238635,
    '121': 98,
    '122': 14,
    '124': 3,
    '125': 8,
    '126': 4,
    '127': 4,
    '128': 3,
    '129': 5,
    '205': 467298,
    '206': 6929,
    '207': 263942,
    '216': 106399,
    '508': 5888,
    '509': 104830,
    '702': 51878,
    '853': 37148,
    '301': 4
}

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.333263Z","iopub.status.idle":"2024-02-18T09:40:50.333629Z","shell.execute_reply.started":"2024-02-18T09:40:50.333459Z","shell.execute_reply":"2024-02-18T09:40:50.333476Z"}}
# data loader to load the data to the model
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class XXDataset(Dataset):
  '''load csv data with feature name ad first row'''
  def __init__(self, datafile):
    super(XXDataset, self).__init__()
    self.feature_names = []
    self.datafile = datafile
    self.data = []
    self._load_data()


  def _load_data(self):
    print("start load data from: {}".format(self.datafile))
    count = 0
    with open(self.datafile) as f:
      #self.feature_names = f.readline().strip().split(',')[2:]
      self.feature_names = use_columns
      for line in f:
        if count == 0:### skip header
          count += 1###
          continue#####
        count += 1
        line = line.strip().split(',')
        line = [int(v) for v in line]
        self.data.append(line)
        if count==1000000:                ################################control number of rows
            break
    print("load data from {} finished".format(self.datafile))

  def __len__(self, ):
    return len(self.data)

  def __getitem__(self, idx):
    line = self.data[idx]
    pv = line[0]
    buy = line[1]
    features = dict(zip(self.feature_names, line[2:]))
    return pv, buy, features



# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.338867Z","iopub.status.idle":"2024-02-18T09:40:50.339479Z","shell.execute_reply.started":"2024-02-18T09:40:50.339198Z","shell.execute_reply":"2024-02-18T09:40:50.339223Z"}}
def get_dataloader(filename, batch_size, shuffle):
  data = XXDataset(filename)
  loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
  return loader

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.341418Z","iopub.status.idle":"2024-02-18T09:40:50.341950Z","shell.execute_reply.started":"2024-02-18T09:40:50.341665Z","shell.execute_reply":"2024-02-18T09:40:50.341722Z"}}
batch_size=1
train_dataloader = get_dataloader(train_path,
                                  batch_size,
                                  shuffle=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.343940Z","iopub.status.idle":"2024-02-18T09:40:50.344413Z","shell.execute_reply.started":"2024-02-18T09:40:50.344184Z","shell.execute_reply":"2024-02-18T09:40:50.344205Z"}}


val_dataloader = get_dataloader(val_path,
                                  batch_size,
                                 shuffle=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.345614Z","iopub.status.idle":"2024-02-18T09:40:50.346035Z","shell.execute_reply.started":"2024-02-18T09:40:50.345839Z","shell.execute_reply":"2024-02-18T09:40:50.345858Z"}}
#next(iter(train_dataloader))

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.347104Z","iopub.status.idle":"2024-02-18T09:40:50.347468Z","shell.execute_reply.started":"2024-02-18T09:40:50.347329Z","shell.execute_reply":"2024-02-18T09:40:50.347343Z"}}
###### first without modefication
import torch
from torch import nn


class Tower2(nn.Module):
  def __init__(self,
               input_dim: int,
               dims=[128, 64, 32],
               drop_prob=[0.1, 0.3, 0.3]):
    super(Tower2, self).__init__()
    self.dims = dims
    self.drop_prob = drop_prob
    self.layer = nn.Sequential(nn.Linear(input_dim, dims[0]), nn.ReLU(),
                               nn.Dropout(drop_prob[0]),
                               nn.Linear(dims[0], dims[1]), nn.ReLU(),
                               nn.Dropout(drop_prob[1]),
                               nn.Linear(dims[1], dims[2]), nn.ReLU(),
                               nn.Dropout(drop_prob[2]))

  def forward(self, x):
    
    x = torch.flatten(x, start_dim=1)
    
    x = self.layer(x)
    
    return x 
    
    
    
    
    

class Tower(nn.Module):
    def __init__(self,
               input_dim: int,
               dims=[128, 64, 32],
               drop_prob=[0.1, 0.3, 0.3],conv_dim = 1):
        super(Tower, self).__init__()
        self.dims = dims
        self.drop_prob = drop_prob
        self.kernal_size = 5
        self.layer = nn.Sequential(nn.Linear(input_dim, dims[1]), nn.ReLU(),
                               nn.Dropout(drop_prob[0]),
                             
                               nn.Linear(dims[1], dims[2]), nn.ReLU(),
                               nn.Dropout(drop_prob[2])
                             )
                                  
                            
        ## conv1D
        self.conv = nn.Sequential(
                   
             
                     torch.nn.Conv1d(conv_dim,64,7, stride=1, padding=3), nn.ReLU(),
                               nn.Dropout(drop_prob[1]),
                     torch.nn.Conv1d(64,32,7, stride=1, padding=3), nn.ReLU(),
                               nn.Dropout(drop_prob[1]),
                     torch.nn.Conv1d(32, conv_dim,7, stride=1, padding=3) )

    def forward(self, x):
       
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim=32):
        super(Attention, self).__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias=False)
        self.k_layer = nn.Linear(dim, dim, bias=False)
        self.v_layer = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        Q = self.q_layer(inputs)
        K = self.k_layer(inputs)
        V = self.v_layer(inputs)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        a = self.softmax(a)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
        return outputs


class AITM(nn.Module):
    def __init__(self,
               feature_vocabulary: dict[str, int],
               embedding_size: int,
               tower_dims=[128, 64, 32],
               drop_prob=[0.1, 0.3, 0.3],batch_size=1):
        super(AITM, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.ModuleDict()
        self.__init_weight()
        self.tower_dims = tower_dims
        self.drop_prob = drop_prob
        self.tower_input_size = len(feature_vocabulary) * embedding_size
        
        # self.click_tower = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.click_tower =Tower(self.tower_input_size, self.tower_dims, self.drop_prob,batch_size)
        self.conversion_tower = Tower2(self.tower_input_size, tower_dims, drop_prob)
        self.attention_layer = Attention(tower_dims[-1])

        self.info_layer = nn.Sequential(nn.Linear(tower_dims[-1], 32), nn.ReLU(),
                                    nn.Dropout(drop_prob[-1]))

        self.click_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                     nn.Sigmoid())
        self.conversion_layer = nn.Sequential(nn.Linear(tower_dims[-1], 1),
                                          nn.Sigmoid())

    def __init_weight(self, ):
        for name, size in self.feature_vocabulary.items():
            emb = nn.Embedding(size, self.embedding_size)
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x):
        feature_embedding = []
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name])
            feature_embedding.append(embed)
        feature_embedding = torch.cat(feature_embedding, 1)
        tower_click = self.click_tower(feature_embedding)

        tower_conversion = torch.unsqueeze(
        self.conversion_tower(feature_embedding), 1)

        info = torch.unsqueeze(self.info_layer(tower_click), 1)

        ait = self.attention_layer(torch.cat([tower_conversion, info], 1))

        click = torch.squeeze(self.click_layer(tower_click), dim=1)
        conversion = torch.squeeze(self.conversion_layer(ait), dim=1)
        
        return click, conversion

    def loss(self,
           click_label,
           click_pred,
           conversion_label,
           conversion_pred,
           constraint_weight=0.6,
           device="gpu:1"):
        click_label = click_label.to(device)
        conversion_label = conversion_label.to(device)

        click_loss = nn.functional.binary_cross_entropy(click_pred, click_label)
        conversion_loss = nn.functional.binary_cross_entropy(
        conversion_pred, conversion_label)

        label_constraint = torch.maximum(conversion_pred - click_pred,
                                     torch.zeros_like(click_label))
        
        constraint_loss = torch.sum(label_constraint)
        
        loss = click_loss + conversion_loss + constraint_weight * constraint_loss
        return loss

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.349321Z","iopub.status.idle":"2024-02-18T09:40:50.349794Z","shell.execute_reply.started":"2024-02-18T09:40:50.349560Z","shell.execute_reply":"2024-02-18T09:40:50.349578Z"}}
# super parameter

embedding_size = 5
learning_rate = 0.0001
total_epoch = 3################## it was 10 epoch
earlystop_epoch = 1
load_model=False

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.351517Z","iopub.status.idle":"2024-02-18T09:40:50.352021Z","shell.execute_reply.started":"2024-02-18T09:40:50.351776Z","shell.execute_reply":"2024-02-18T09:40:50.351798Z"}}
train_loss_list=[]

def train():
  # train_dataloader = get_dataloader(train_path,
  #                                   batch_size,
  #                                   shuffle=True)
  # dev_dataloader = get_dataloader(val_path,
  #                                 batch_size,
  #                                 shuffle=True)
  model = AITM(vocabulary_size, embedding_size)
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=1e-6)



  if load_model:
    load_checkpoint(torch.load("/kaggle/working/checkpoint.pth.tar"), model, optimizer)
  model.to(device)
  best_acc = 0.0
  earystop_count = 0
  best_epoch = 0
  for epoch in range(total_epoch):
    total_loss = 0.
    nb_sample = 0
    if epoch == 2:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),'loss': loss}
        # Try save checkpoint
        save_checkpoint(checkpoint)    
    # train
    model.train()
    for step, batch in enumerate(train_dataloader): # step => counter
      #print(batch) # [click , convertion , {features}]
      # [tensor([0]), tensor([0]), {'101': tensor([0]), '121': tensor([0]), '122': tensor([0]), '124': tensor([0]), '125': tensor([0]), '126': tensor([0]), '127': tensor([0]), '128': tensor([0]), '129': tensor([0]), '205': tensor([0]), '206': tensor([0]), '207': tensor([0]), '216': tensor([0]), '508': tensor([0]), '509': tensor([0]), '702': tensor([0]), '853': tensor([0]), '301': tensor([0])}]
      click, conversion, features = batch
      for key in features.keys():
        features[key] = features[key].to(device)
      click_pred, conversion_pred = model(features)
      loss = model.loss(click.float(),
                        click_pred,
                        conversion.float(),
                        conversion_pred,
                        device=device)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.cpu().detach().numpy()
      nb_sample += click.shape[0]
      if step % 10000 == 0:
        print('[%d] Train loss on step %d: %.6f' %
              (nb_sample, (step + 1), total_loss / (step + 1)))
        train_loss_list.append(total_loss / (step + 1))

 # validation
    print("start validation...")
    click_pred = []
    click_label = []
    conversion_pred = []
    conversion_label = []
    model.eval()
    for step, batch in enumerate(val_dataloader):
      click, conversion, features = batch
      for key in features.keys():
        features[key] = features[key].to(device)

      with torch.no_grad():
        click_prob, conversion_prob = model(features)

      click_pred.append(click_prob.cpu())
      conversion_pred.append(conversion_prob.cpu())

      click_label.append(click)
      conversion_label.append(conversion)

    click_auc = cal_auc(click_label, click_pred)
    conversion_auc = cal_auc(conversion_label, conversion_pred)
    print("Epoch: {} click_auc: {} conversion_auc: {}".format(
        epoch + 1, click_auc, conversion_auc))

    acc = click_auc + conversion_auc
    torch.save(model.state_dict(), model_file)
    
    fpr_click, tpr_click, _ = auc_curve(click_label, click_pred)
    fpr_conversion, tpr_conversion, _ = auc_curve(conversion_label, conversion_pred)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k-')
    plt.plot(fpr_click, tpr_click, label='Click (AUC = {:.3f})'.format(click_auc))
    plt.plot(fpr_conversion, tpr_conversion, label='Conversion (AUC = {:.3f})'.format(conversion_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

  

def test():
  print("Start Test ...")
  test_loader = get_dataloader(test_path,
                               batch_size=batch_size,
                               shuffle=False)
  model = AITM(vocabulary_size, 5)
  model.load_state_dict(torch.load(model_file))
  model.eval()
  click_list = []
  conversion_list = []
  click_pred_list = []
  conversion_pred_list = []
  for i, batch in enumerate(test_loader):
    if i % 100:
      sys.stdout.write("test step:{}\r".format(i))
      sys.stdout.flush()
    click, conversion, features = batch
    with torch.no_grad():
      click_pred, conversion_pred = model(features)
    click_list.append(click)
    conversion_list.append(conversion)
    click_pred_list.append(click_pred)
    conversion_pred_list.append(conversion_pred)
  click_auc = cal_auc(click_list, click_pred_list)
  conversion_auc = cal_auc(conversion_list, conversion_pred_list)
  print("Test Resutt: click AUC: {} conversion AUC:{}".format(
      click_auc, conversion_auc))
def auc_curve(label: list, pred: list):
  label = torch.cat(label)
  pred = torch.cat(pred)
  label = label.detach().numpy()
  pred = pred.detach().numpy()
  auc = roc_auc_score(label, pred, labels=np.array([0.0, 1.0]))
  fpr, tpr, _ = roc_curve(label, pred)
  return fpr, tpr, _

  fpr_click, tpr_click, _ = auc_curve(click_list, click_pred_list)
  fpr_conversion, tpr_conversion, _ = auc_curve(conversion_list, conversion_pred_list)

  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k-')
  plt.plot(fpr_click, tpr_click, label='Click (AUC = {:.3f})'.format(click_auc))
  plt.plot(fpr_conversion, tpr_conversion, label='Conversion (AUC = {:.3f})'.format(conversion_auc))
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend(loc='best')
  plt.show()






def cal_auc(label: list, pred: list):
  label = torch.cat(label)
  pred = torch.cat(pred)
  label = label.detach().numpy()
  pred = pred.detach().numpy()
  auc = roc_auc_score(label, pred, labels=np.array([0.0, 1.0]))
  return auc

def auc_curve(label: list, pred: list):
  label = torch.cat(label)
  pred = torch.cat(pred)
  label = label.detach().numpy()
  pred = pred.detach().numpy()
  auc = roc_auc_score(label, pred, labels=np.array([0.0, 1.0]))
  fpr, tpr, _ = roc_curve(label, pred)
  return fpr, tpr, _


def save_checkpoint(state,filename='/kaggle/working/checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state,filename)

    

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    loss = checkpoint['loss']

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.353048Z","iopub.status.idle":"2024-02-18T09:40:50.353448Z","shell.execute_reply.started":"2024-02-18T09:40:50.353247Z","shell.execute_reply":"2024-02-18T09:40:50.353266Z"}}
# Start the timer
#kernel 7-7-5
from timeit import default_timer as timer 
start_time = timer()
train()
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.354704Z","iopub.status.idle":"2024-02-18T09:40:50.355097Z","shell.execute_reply.started":"2024-02-18T09:40:50.354896Z","shell.execute_reply":"2024-02-18T09:40:50.354916Z"}}
# Start the timer
from timeit import default_timer as timer 
start_time = timer()
test()
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.357719Z","iopub.status.idle":"2024-02-18T09:40:50.358190Z","shell.execute_reply.started":"2024-02-18T09:40:50.357968Z","shell.execute_reply":"2024-02-18T09:40:50.357989Z"}}
# Start the timer
#kernel 15
from timeit import default_timer as timer 
start_time = timer()
train()
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.359449Z","iopub.status.idle":"2024-02-18T09:40:50.359873Z","shell.execute_reply.started":"2024-02-18T09:40:50.359643Z","shell.execute_reply":"2024-02-18T09:40:50.359663Z"}}
# Start the timer
#kernel 3
from timeit import default_timer as timer 
start_time = timer()
train()
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.361581Z","iopub.status.idle":"2024-02-18T09:40:50.362037Z","shell.execute_reply.started":"2024-02-18T09:40:50.361824Z","shell.execute_reply":"2024-02-18T09:40:50.361847Z"}}
# Start the timer
#kernel 9
from timeit import default_timer as timer 
start_time = timer()
train()
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.363577Z","iopub.status.idle":"2024-02-18T09:40:50.363972Z","shell.execute_reply.started":"2024-02-18T09:40:50.363811Z","shell.execute_reply":"2024-02-18T09:40:50.363826Z"}}
# Start the timer
#kernel 5
from timeit import default_timer as timer 
start_time = timer()
train()
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %% [code] {"execution":{"iopub.status.busy":"2024-02-18T09:40:50.364934Z","iopub.status.idle":"2024-02-18T09:40:50.365244Z","shell.execute_reply.started":"2024-02-18T09:40:50.365097Z","shell.execute_reply":"2024-02-18T09:40:50.365110Z"}}
# Start the timer
from timeit import default_timer as timer 
start_time = timer()
test()
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %% [code]


# %% [code]
