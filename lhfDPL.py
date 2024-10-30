# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:29:55 2022
use met value  to regress the flux   

感热通量回归



@author: mayubin 
2022-6-9
2023-2-21
2023-3-9
"""

#%%
#shouxian shezhi train he test ji he 


import sklearn
import xarray as xr

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os
import pandas as pd
import math
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



data = np.loadtxt('metz_hr_3.txt')


#有些nan 在数据中，这不利于数据处理，得清洗数据



u = - np.abs(data[:,3])* np.sin(3.14*data[:,4]/180)
v = - np.abs(data[:,3])* np.cos(3.14*data[:,4]/180)

data[:,0] = u
data[:,1] = v

qs_obs = data[:,10]
q_obs=data[:,11] 

hsb_mean_ci = (data[:,12]+data[:,13])/2
data[:,33]  = hsb_mean_ci
hlb_mean_ci = (data[:,15]+data[:,16])/2
data[:,34]  = hlb_mean_ci
#tau_mean_ci = (data[:,16]+data[:,17])/2
#data[:,25]  = tau_mean_ci 

hlb = data[:,17]
hsb = data[:,14]


tau_i = data[:,20]
tau_b = data[:,21]


#%%
'''
The columns are as follows:
     
0 --> u   Year=x(:,1);%year of the field program   
1 --> v  jdy=x(:,2);%julian day at beginning of time average
2   ushp=x(:,3);%doppler log, SCS (m/s)
3   U=x(:,4);%true wind,ETL sonic (m/s)
4   dir=x(:,5);%true wind direction, ETL sonic (deg)
5   urel=x(:,6);%relative wind speed, ETL (m/s)
6  reldir=x(:,7);%relative wind direction (from),clockwise rel ship's bow, ETL
     sonic (deg)
7  head=x(:,8);%ship heading, deg clockwise rel north, SCS laser ring gyro
     (deg)
8   tsnk=x(:,9);%sea snake temperature, ETL, 0.05 m depth (C)
9  ta=x(:,10);%air temperature, ETL (C)   
10   qse=x(:,11);%sea surface specific humidity, from snake (g/kg)
11   qa=x(:,12);%air specific humidity, ETL (g/kg)
12   hsc=x(:,13);%sensible heat flux, covariance, ETL sonic anemometer(W/m^2)
13   hsib=x(:,14);%sensible heat flux, ID, ETL sonic anemometer(W/m^2)
14   hsb=x(:,15);%bulk sensible heat flux, (W/m^2)
15   hlc=x(:,16);%latent heat flux, covariance, (W/m^2)
16   hlib=x(:,17);%latent heat flux, ID, (W/m^2)
17   hlb=x(:,18);%bulk latent heat flux, W/m^2 (includes Webb et al. correction)
18   taucx=x(:,19);%covariance streamwise stress, ETL sonic anemometer (N/m^2)
19   taucy=x(:,20);%covariance cross-stream stress, ETL sonic anemometer (N/m^2)
20   tauib=x(:,21);%ID streamwise stress, ETL sonic anemometer (N/m^2)
21   taub=x(:,22);%bulk wind stress along mean wind, (N/m^2)
22   rs=x(:,23);%downward solar flux, ETL units (W/m^2)
23   rl=x(:,24);%downward IR flux,  ETL units (W/m^2)
24   org=x(:,25);%rainrate, ETL STI optical rain gauge, uncorrected (mm/hr)
25   J=x(:,26);%ship plume contamination index
26   sigoph=x(:,27);%standard deviation of ophir fast hygrometer clear channel
27   tiltx=x(:,28);%flow tilt at ETL sonic anemometer, earth frame
28   Jm=x(:,29);%ship maneuver index
29   ct=x(:,30);%ct^2 (K^2/m^.667)
30   cq=x(:,31);%cq^2 ((g/kg)^2/m^.667) 
31  cu=x(:,32);%cu^2 ((m/s)^2/m^.667)
32   cw=x(:,33);%cw^2 ((m/s)^2/m^.667)
33  --> hs_mean hrain=x(:,34);%rain heat flux,Gosnell et al 1995, JGR, 18437-18442, (W/m^2)
34  hlwebb=x(:,35);%correction to measured latent heat flux, Webb et al.
     1980,QJRMS, 85-100--> hl_mean
35   lat=x(:,36);%latitude, deg  (SCS pcode)
36   lon=x(:,37);%longitude, deg (SCS pcode)
37  zu_etl=x(:,38);%height of mean wind sensor, 17.7 m
38   zt_etl=x(:,39);%height of mean air temperature sensor, 15.5 m
39   zq_etl=x(:,40);%height of mean air humidity sensor, 15.5 m
   %*****   ships imet and scs data
40   sog=x(:,41);%speed over ground, SCS gps, (m/s)
41   U_scs=x(:,42); %true wind speed, imet propvane anemometer (m/s)
42   dir_scs=x(:,43);%true wind direction (from),clockwise rel north, imet,(deg)
43   cog=x(:,44);%%course over ground, SCS gps, (m/s)
44   tsg=x(:,45);%tsg water temperature, 5 m depth, (C)
45   ta_im=x(:,46);%imet air temperature (C) 
46   qs_tsg=x(:,47);%imet bulk water specific humidity (g/kg) 
47   qa_im=x(:,48);%imet air specific humidity, (g/kg)

'''
feature = [0,1,10,11]
feature_for_train=[0,1,10,11,34]
#潜热的因子
#feature = [0,1,6,7,8,9,19]
#feature_for_train=[0,1,6,7,8,9,19,33]



feature_count = np.size(feature)


data_less = data[:,feature_for_train]

data_less = pd.DataFrame(data_less)
data_less = data_less.dropna()

data_train1 = data_less.values[[i for i in range(len(data_less)) if i % 10 != 0],:]
data_test1 = data_less.values[[i for i in range(len(data_less)) if i % 10 == 0],:]


data_train = data_train1[:,:]
data_test = data_test1[:,:-1]
data_train = pd.DataFrame(data_train)
data_test =pd.DataFrame(data_test)

data_train.to_csv('train.csv',index=False)
data_test.to_csv('test.csv',index=False)
#还是保存成csv吧。就先回归潜热吧

#%%
#xiamina jiu kaishi jinxin xunlian 

tr_path = 'train.csv'  # path to training data
tt_path = 'test.csv'   # path to testing data



myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# # **Some Utilities**

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    #plt.ylim(0.0, 5000.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    #plt.show()
    plt.savefig(r'train.png')


def plot_pred(dv_set, model, device, lim=400, preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    #plt.show()
    plt.savefig(r'preds.png')



point_count = 8

# # **Preprocess**
# 
# We have three kinds of datasets:
# * `train`: for training
# * `dev`: for validation
# * `test`: for testing (w/o target value)

# ## **Dataset**
# 
# The `myDataset` below does:
# * read `.csv` files
# * extract features
# * split `covid.train.csv` into train/dev sets
# * normalize features
# 

class myDataset(Dataset):
    ''' Dataset for loading and preprocessing the  dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, :].astype(float)
        
        if not target_only:
            feats = list(range(feature_count))
        else:
           feats = list(range(feature_count))# feats = list(range(40))
           # feats.extend([57,75])# TODO

        if mode == 'test':
            # Testing data    
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
      #  self.data[:, :] =  (self.data[:, :] - self.data[:, :].mean(dim=0, keepdim=True))   / self.data[:, :].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


# ## **DataLoader**
# 
# A `DataLoader` loads data from a given `Dataset` into batches.
# 

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = myDataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


# # **Deep Neural Network**
# 
# `NeuralNet` is an `nn.Module` designed for regression.
# The DNN consists of 2 fully-connected layers with ReLU activation.
# This module also included a function `cal_loss` for calculating loss.
# 
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim,point_count):
        super(NeuralNet, self).__init__()
        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L2 regularization here
        return self.criterion(pred, target)


# # **Train/Dev/Test**

# ## **Training**

# In[7]:


def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 10000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        #print('epoch = ',epoch,',loss = ',dev_mse)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


# ## **Validation**

def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


# ## **Testing**

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds



#%%
#关键的地方了
# # **Setup Hyper-parameters**
# 
# `config` contains hyper-parameters for training and the path to save your model.

device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                 # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 4000,                # maximum number of epochs
    'batch_size': 500,               # mini-batch size for dataloader
    'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
   # 'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
     #   'lr': 0.001,                 # learning rate of SGD
    #    'momentum': 0.09              # momentum for SGD
    },
    'early_stop': 1000,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}


# # **Load data and model**

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim,point_count).to(device)  # Construct model and move to device

# # **Start Training!**

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim,point_count).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)

plot_pred(dv_set, model, device)  # Show prediction on the validation set


# # **Testing**
# The predictions of your model on testing set will be stored at `pred.csv`.

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  
save_pred(preds, 'pred_lhf.csv')         # save prediction file to pred.csv



print('dp lhf 误差分析')
print('rmse = ',np.sqrt(model_loss))
print('corrcoef = ',np.corrcoef(data_test1[:,-1],preds)[0,1])
print('std = ',np.std(preds))
#%%

def count_nonnan(x):
    #统计数组中非nan的个数
    import numpy as np
    return  np.count_nonzero(~np.isnan(x))
     

def mse(y_test,y_predict):
    #原生实现
    # 衡量线性回归的MSE 、 RMSE、 MAE、r2
    mean_y_test = np.nanmean(y_test)
    mse = np.nansum((y_test - y_predict) ** 2) /count_nonnan(y_test)
    rmse = math.sqrt(mse)
    rmse_mean = rmse/mean_y_test
    print('',"\n mse:",mse," \n rmse:",rmse,"\n rmse/mean:",rmse_mean,'\n')



print('latent heat')
mse(hlb_mean_ci,hlb)

print('sensible heat')
mse(hsb_mean_ci,hsb)

print('momentum flux')
mse(tau_i,tau_b )


data_test2 = data[6001:7200,:]
data_test2 = data_test2[:,feature_for_train]
data_test2 =pd.DataFrame(data_test2)
data_test2 = data_test2.dropna()


print('latent heat deeplearning')
y2 = mse(data_test2.values[:,-1], preds)

figure()
plt.plot(hlb[6001:7200],'r.',label = 'COARE')
plt.plot(hlb_mean_ci[6001:7200],'k.',label = 'obs')
plt.plot(preds,'b.',label = 'DP')
plt.xlabel('hours')
plt.ylabel('Wm^-2')
plt.legend()
plt.title('latent heat on test set ')
plt.show()


figure()
plt.stem(preds-hlb_mean_ci[6001:7200],'r.',label = 'DP - obs')
plt.stem(hlb[6001:7200]-hlb_mean_ci[6001:7200],'b.',label = 'COARE - obs')

plt.legend()
plt.show()

figure()
plt.plot(data[6001:7200,15],'r.',label = 'covariance')
plt.plot(data[6001:7200,16],'b.',label = 'ID')
plt.legend()
plt.show()
#%%

#构建潜热数据集
time_1 = data.u10.shape[0]
filename_analysis = r"H:\fluxdeepl\shf_dp_combine\uvttr_ec_low.nc"
ap_path = "apply.csv"
data=xr.open_dataset(filename_analysis)
data1 = xr.open_dataset(r'td_ec_low.nc')
u10 = np.ones((time_1,72,144),dtype=float)
v10 = np.ones((time_1,72,144),dtype=float)
t2m = np.ones((time_1,72,144),dtype=float)
sst = np.ones((time_1,72,144),dtype=float)
q = np.ones((time_1,72,144),dtype=float)
qs = np.ones((time_1,72,144),dtype=float)
td = np.ones((time_1,72,144),dtype=float)
rh = np.ones((time_1,72,144),dtype=float)

lhf_analysis_3D = np.ones((time_1,72,144),dtype=float)

#这里的温度是K温标
#np.nanmean(data.ssrdc)/30/24/60
#Out[211]: 427.8178703703704
#这里的是J/month


for i in range(time_1-3):
    print('time = ',i, 'total =', time_1)
    u10[i,:,:] = data.u10[i,0,:,:].data
    v10[i,:,:] = data.v10[i,0,:,:].data
    t2m[i,:,:] = data.t2m[i,0,:,:].data-273.15
    sst[i,:,:] = data.sst[i,0,:,:].data-273.15
#这里需要改成湿度的变量，海表比湿，大气比湿！
    td[i,:,:]  = data1.d2m[i,0,:,:].data-273.15

    #qs[i,:,:] = data.qs[i,0,:,:].data  
    #q[i,:,:]  = data.q[i,0,:,:].data
#EC 只有露点温度，没有比湿，露点温度，温度算比湿，再用海表温度算海表饱和比湿：

#rh = 100 - 5(T - Td)
#    rh[i,:,:] = (100 - 5*(t2m[i,:,:] - td[i,:,:]))*0.01#有些是小于0了，这个公式会导致问题。。
#Magnus-Tetens 方法
    para_a = 17.27;para_b = 237.7 # t 和 td 都是C 温标
    rh[i,:,:] = np.exp(para_a*td[i,:,:]/(para_b+td[i,:,:]) - para_a*t2m[i,:,:]/(para_b+t2m[i,:,:]))
    rh[rh<0] = 0
#qs = qs(sst)
    qs[i,:,:] = 0.98*6.112*np.exp((sst[i,:,:]*17.62)/(sst[i,:,:]+243.12))
#rh = q/qs --q = qs * rh 
    q[i,:,:]  = qs[i,:,:] * rh[i,:,:]
    

  
    u10_temp   = u10[i,:,:].reshape(72*144,-1)
    v10_temp   = v10[i,:,:].reshape(72*144,-1)
    t2m_temp   = t2m[i,:,:].reshape(72*144,-1)   
    sst_temp   = sst[i,:,:].reshape(72*144,-1)
    q_temp     = q[i,:,:].reshape(72*144,-1)
    qs_temp    = qs[i,:,:].reshape(72*144,-1)
    
#    rdc_temp   = rdc[i,:,:].reshape(72*144,-1)
#    quecebuqi(sst_temp)  
#一个关键的问题是不能有缺测，这样陆地也得计算
    sst_temp[np.isnan(sst_temp)] = np.nanmean(sst_temp)
    q_temp[np.isnan(q_temp)] = np.nanmean(q_temp)
    qs_temp[np.isnan(qs_temp)] = np.nanmean(qs_temp)
    
#这里需要的比湿的数据还没加上
    data_merge = np.hstack((u10_temp,v10_temp,qs_temp,q_temp))#sst_temp,t2m_temp
    data_merge = pd.DataFrame(data_merge)
    data_merge.to_csv('apply.csv',index=False)
    #数组太大了，导致这里的程序运行非常缓慢！！！硬盘写入速度很慢


    ap_set = prep_dataloader(ap_path, 'test', config['batch_size'])

    lhf_analysis = test(ap_set,model,device)

    lhf_analysis_3D[i,:,:] = lhf_analysis.reshape((72,144))
    
    #程序到这会断下，手动运行下一个段！

#%%
#添加一些判断，
#海冰时候不计算
#陆地不计算，在sst 为nan时候，data = nan 
#在sst小于海冰值的时候 data = nan
#对经度纬度进行循环

'''
对海温的修正
'''

for t in range(lhf_analysis_3D.shape[0]):#
    print(t)
    #获取当前时间层海温数据
    data_sst_now = data.sst.data[t,0,:,:]
    #获得当前时间层海温nan索引
    ind_nan_sst = np.where(np.isnan(data_sst_now))
    #海温小于冰点的索引
    ind_ice_sst = np.where(data_sst_now <  273.15)
    #获得当前时间层潜热通量
    lhf_analysis_3D_now = lhf_analysis_3D[t,:,:]
    #赋值nan
    lhf_analysis_3D_now[ind_nan_sst] = np.nan
    lhf_analysis_3D_now[ind_ice_sst] = np.nan
    #再写进三维的数组里
    lhf_analysis_3D[t,:,:] = lhf_analysis_3D_now
    
    


#%%
def write_to_nc(data,file_name_path):
    import netCDF4 as nc
    time = np.linspace(1,1002,1002)
    lonS=np.linspace(0,357.5,144)
    latS=np.linspace(-88.75,88.75,72)
    da=nc.Dataset(file_name_path,'w',format='NETCDF4')
    da.createDimension('time', 1002)
    da.createDimension('lon',144) #创建坐标点
    da.createDimension('lat',72) #创建坐标点
    da.createVariable('time', 'f', ("time"))
    da.createVariable("lon",'f',("lon")) #添加coordinates 'f'为数据类型，不可或缺
    da.createVariable("lat",'f',("lat")) #添加coordinates 'f'为数据类型，不可或缺
    da.variables['lat'][:]=latS  #填充数据
    da.variables['lon'][:]=lonS  #填充数据
    da.variables['time'][:] = time
    da.createVariable(write_to_nc.__code__.co_varnames[0],'f8',('time','lat','lon')) #创建变量，shape=(721,1440) 'f'为数据类型，不可或缺
    da.variables[write_to_nc.__code__.co_varnames[0]][:]=data  #填充数据 
    da.close()
#os.remove('tau_analysis_3d.nc')
write_to_nc(lhf_analysis_3D,'lhf_analysis_3d.nc')
write_to_nc(qs,'qs.nc')
write_to_nc(q,'q.nc')
write_to_nc(rh,'rh.nc')
#相关变量



##========================================================================
filename2 = r'H:\fluxdeepl\oaflux\latent_low.nc'

obs=xr.open_dataset(filename2)

latent = obs.lhtfl.data
latent_mean_year = np.nanmean(latent,axis = 0)
lhf_analysis_3D_mean = np.nanmean(lhf_analysis_3D[995-11*12-288:995-11*12,:,:],axis = 0)
lhf_delta = lhf_analysis_3D_mean-latent_mean_year
write_to_nc(lhf_delta,'lhf_delta.nc')
#%%
##========================================================================
write_to_nc(latent_mean_year,'lhf_ec_mean.nc')
write_to_nc(lhf_analysis_3D_mean, 'lhf_analysis_3D_mean.nc')

write_to_nc(sst, 'sst.nc')
#%%

#%%
#%%
#模型参数传递torch->TBF->Fortran


#  计算模型参数总数。 
total = sum(p.numel() for p in model.parameters())
print("总参数个数: %.2f" % (total))

#将pth 的模型参数输出出来

#有几个因子  m = ?
m = len(feature)

#有几层  o = ?
o = ( len(list(model.net)) -1 )//2 #自动获取层数，减去输出层然后线性层和激活函数层的和整除以2

#每层有几个节点 n = ?
n = point_count

#输入层的权重和偏差
w_input =  model.net[0].weight.data.cpu().numpy()
b_input =  model.net[0].bias.data.cpu().numpy()

#print( model.net[0].weight.data.cpu().numpy())
np.savetxt('w_input.txt',w_input,fmt='%f')
np.savetxt('b_input.txt',b_input,fmt='%f')

#中间层的权重和偏差
os.system('rd w_dense.txt ')
os.system('rd b_dense.txt')

for i in range(2,2*o,2):# range(x,y) 不包含y 
#	print(model.net[i].weight.data.cpu().numpy())

	w_dense = model.net[i].weight.data.cpu().numpy()
	b_dense = model.net[i].bias.data.cpu().numpy()

	with open("w_dense.txt","ab") as f:#追加写入模式
		np.savetxt(f,w_dense,fmt = '%f')
	with open("b_dense.txt","ab") as g:
		np.savetxt(g,b_dense,fmt = '%f')


#输出层的权重和误差

#print( model.net[2*o].weight.data.cpu().numpy())

w_output =  model.net[2*o].weight.data.cpu().numpy()
b_output =  model.net[2*o].bias.data.cpu().numpy()

np.savetxt('w_output.txt',w_output,fmt='%f')
np.savetxt('b_output.txt',b_output,fmt='%f')


#向文件写入维度参数
#

note = open ('shuchucanshu.txt','w')
note.write('tbf ')
note.write('\n存放深度学习模型数据，用于输入到TBF中')

note.write('\nm\n')
note.write(str(m))

note.write('\nn\n')
note.write(str(n))

note.write('\no\n')
note.write(str(o))

note.close