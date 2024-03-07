import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
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
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

# Model, optimizer, and loss function
input_dim=30     # how many features are in the dataset or how many input nodes in the input layer
hidden_dim1 = 50
hidden_dim2 = 40 
hidden_dim3 = 30 
output_dim=4    # number of classes

print(input_dim,hidden_dim1,hidden_dim2,hidden_dim3,output_dim)

# Instantiate model
# Check if GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
model.to(device)
# Calculate class weights for handling imbalance
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_y.to_numpy()),
    y=train_y.to_numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Adam optimizer with a different learning rate
criterion = nn.CrossEntropyLoss(weight=class_weights)

print('W:',list(model.parameters())[0].size())
print('b',list(model.parameters())[1].size())
# Training process with early stopping
n_epochs = 600
loss_list = []
best_loss = float('inf')
patience = 20  # Patience for early stopping
trigger_times = 0

# # loss function
# criterion=nn.CrossEntropyLoss()
# learning_rate=0.05
# optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
# n_epochs=600
# loss_list=[]

#n_epochs
for epoch in range(n_epochs):
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    # Early stopping condition
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {current_loss}")
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
#confusion matrix
# [[[4214 2570]
#   [1197 2203]]

#  [[6077 2079]
#   [1427  601]]

#  [[6843  179]
#   [3065   97]]

#  [[6926 1664]
#   [ 803  791]]] 
# 27752/40736=0.68
# Average Precision: 0.33986429430181864
# Average Recall: 0.367801233173527
# Average F1 Score: 0.310386038867453