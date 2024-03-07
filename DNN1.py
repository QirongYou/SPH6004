import torch.nn as nn
import torch
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from IPython.display import display

torch.manual_seed(1121)
df = pd.read_csv('sph6004_assignment1_data_final.csv')

columns_to_use = ['bun_min', 'inr_max', 'gcs_verbal_0.0', 'ph_min', 'admission_age', 'Glucose_Variability', 'gcs_motor_6.0', 'aniongap_max', 'calcium_min', 'weight_admit', 'ph_max', 
    'glucose_max', 'BMI', 'sbp_min', 'albumin_min', 'Oxygenation_Index', 'Acid_Base_Balance_Indicator', 'baseexcess_max', 'dbp_min', 'aniongap_min', 'inr_min', 'ptt_max', 
    'lactate_min', 'spo2_min', 'bilirubin_total_min', 'calcium_max', 'height', 'potassium_max', 'hematocrit_max', 'potassium_max.1']
data=df[columns_to_use]
target=df.iloc[:,0]
 
# 切分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data,target,test_size=0.2,random_state=42)
# Check data types and convert if necessary
bool_columns = train_x.select_dtypes(include=['bool']).columns
train_x[bool_columns] = train_x[bool_columns].astype(float)
bool_columns_test = test_x.select_dtypes(include=['bool']).columns
test_x[bool_columns_test] = test_x[bool_columns_test].astype(float)

class Data(Dataset):
    def __init__(self):
        self.x=torch.from_numpy(train_x.values)
        self.y=torch.from_numpy(train_y.values).long()
        self.len=self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
data_set=Data()


# define batch sizes here 
batch_size = 64
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs).float()
    targets = torch.tensor(targets).long()
    return inputs, targets

trainloader = DataLoader(dataset=data_set, batch_size=batch_size, collate_fn=collate_fn)

# D_in, dimension of input layer
# H , dimension of hidden layer
# D_out, dimension of output layer
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        # Ensure x is a Float tensor
        x = x.float()
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

input_dim=30     # how many features are in the dataset or how many input nodes in the input layer
hidden_dim = 20 # hidden layer size
output_dim=4    # number of classes
print(input_dim,hidden_dim,output_dim)


# Instantiate model
model=Net(input_dim,hidden_dim,output_dim)

print('W:',list(model.parameters())[0].size())
print('b',list(model.parameters())[1].size())


# loss function
criterion=nn.CrossEntropyLoss()
learning_rate=0.05
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
n_epochs=600
loss_list=[]

#n_epochs
for epoch in range(n_epochs):
    if epoch %100==0:
        print(epoch)
    for x, y in trainloader:
      
        #clear gradient 
        optimizer.zero_grad()
        #make a prediction 
        z=model(x)
        #print(z)
        # calculate loss, da Cross Entropy benutzt wird muss ich in den loss Klassen vorhersagen, 
        # also Wahrscheinlichkeit pro Klasse. Das mach torch.max(y,1)[1])
        loss=criterion(z,y)
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()
        
        loss_list.append(loss.data)
        
        
        #print('epoch {}, loss {}'.format(epoch, loss.item()))
# predict the class and give probablity for each class label
x_val = torch.tensor(test_x.values, dtype=torch.float32)
y_val = torch.from_numpy(test_y.values).float()
z=model(x_val)
display(z)
# choose the predicted class to be the one with maximum probablity
yhat=torch.max(z.data,1)
display(yhat)
# get the indicies
y_pred=yhat.indices.detach().numpy()
display(y_pred)
display(y_val)
from sklearn.metrics import multilabel_confusion_matrix
display(multilabel_confusion_matrix(y_val, y_pred))
# #confusion matrix
# [[[3638 3146]
#   [ 732 2668]]

#  [[8139   17]
#   [2021    7]]

#  [[5110 1912]
#   [1713 1449]]

#  [[8121  469]
#   [1078  516]]]
# 29648/40736=0.72
# Average Precision: 0.42638463838039065
# Average Recall: 0.39253143888958947
# Average F1 Score: 0.35759263758276905