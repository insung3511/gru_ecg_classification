# https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/

import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch

EPOCH = 200
KERNEL_SIZE = 3
POOLING_SIZE = 2

DATA_PATH = "./mit_data/"

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# Dataload part
le = preprocessing.LabelEncoder()

record_list = []
pickle_input = dict()
X, y = [], []

print("[INFO] Read records file from ", DATA_PATH)
with open(DATA_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

for i in range(len(record_list)):
    temp_path = DATA_PATH + "mit" + record_list[i] + ".pkl"
    with open(temp_path, 'rb') as f:
        pickle_input = pickle.load(f)
        X.append(pickle_input[0])

        for j in range(len(pickle_input[1][i])):
            check_ann = pickle_input[1][i][j]
            temp_ann_list = list()
            if check_ann == "N":            # Normal
                temp_ann_list.append(0)
            elif check_ann == "S":          # Supra-ventricular
                temp_ann_list.append(1)
            elif check_ann == "V":          # Ventricular
                temp_ann_list.append(2)
            elif check_ann == "F":          # False alarm
                temp_ann_list.append(3)
            else:                           # Unclassed 
                temp_ann_list.append(4)
            y.append(temp_ann_list)


print("[SIZE] X length : {}, y length : {}".format(len(X), len(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

print("[SIZE]\t\tTrain X size : {}, Train y size : {}\n\t\tTest X size : {}, Test y size : {}"\
        .format(len(X_train), len(y_train), len(X_test), len(y_test)))

train_dataloadloader = DataLoader(X_train)
test_dataloader = DataLoader(X_test)

# Init
loss_fun = nn.MSELoss().to(device=device)
criterion = CrossEntropyLoss().to(device=device)

# First Part
conv1 = nn.Conv1d(in_channels=428, out_channels=32, kernel_size=9)
bn1 = nn.BatchNorm1d(32)
relu1 = nn.ReLU()

# Second Part
conv2a = nn.Conv1d(in_channels=93, out_channels=64, kernel_size=KERNEL_SIZE)
conv2b = nn.Conv1d(in_channels=93, out_channels=64, kernel_size=KERNEL_SIZE)
bn2 = nn.BatchNorm1d(64)
relu2 = nn.ReLU()
maxpool2 = nn.MaxPool1d(64, stride=64)

# Third part
conv3a = nn.Conv1d(in_channels=47, out_channels=64, kernel_size=KERNEL_SIZE)
conv3b = nn.Conv1d(in_channels=47, out_channels=64, kernel_size=KERNEL_SIZE)
bn3 = nn.BatchNorm1d(64)
relu3 = nn.ReLU()
maxpool3 = nn.MaxPool1d(64, stride=128)

# Fourth part
conv4a = nn.Conv1d(in_channels=24, out_channels=128, kernel_size=KERNEL_SIZE)
conv4b = nn.Conv1d(in_channels=24, out_channels=128, kernel_size=KERNEL_SIZE)
bn4 = nn.BatchNorm1d(128)
relu4 = nn.ReLU()
maxpool4 = nn.MaxPool1d(128, stride=128)

# Fiveth part
conv5a = nn.Conv1d(in_channels=12, out_channels=128, kernel_size=KERNEL_SIZE)
conv5b = nn.Conv1d(in_channels=12, out_channels=128, kernel_size=KERNEL_SIZE)
bn5 = nn.BatchNorm1d(128)
relu5 = nn.ReLU()
maxpool5 = nn.MaxPool1d(128, stride=256)

# Sixth part
conv6a = nn.Conv1d(in_channels=6, out_channels=256, kernel_size=KERNEL_SIZE)
conv6b = nn.Conv1d(in_channels=6, out_channels=256, kernel_size=KERNEL_SIZE)
bn6 = nn.BatchNorm1d(256)
relu6 = nn.ReLU()
maxpool6 = nn.MaxPool1d(256, stride=256)

# Seventh part
conv7a = nn.Conv1d(in_channels=3, out_channels=256, kernel_size=KERNEL_SIZE)
conv7b = nn.Conv1d(in_channels=3, out_channels=256, kernel_size=KERNEL_SIZE)
bn7 = nn.BatchNorm1d(256)
relu7 = nn.ReLU()
maxpool7 = nn.MaxPool1d(256, stride=256)

first_input_layer = nn.Sequential(
    conv1, bn1, relu1
)

LFEM1 = nn.Sequential(
    conv2a, conv2b, bn2, relu2, maxpool2
)

LFEM2 = nn.Sequential(
    conv3a, conv3b, bn3, relu3, maxpool3
)

LFEM3 = nn.Sequential(
    conv4a, conv4b, bn4, relu4, maxpool4
)

LFEM4 = nn.Sequential(
    conv5a, conv5b, bn5, relu5, maxpool5
)

LFEM5 = nn.Sequential(
    conv6a, conv6b, bn6, relu6, maxpool6
)

LFEM6 = nn.Sequential(
    conv7a, conv7b, bn7, relu7, maxpool7
)

# Optim
first_optimizer = optim.Adam(first_input_layer.parameters())
lfem1_optimizer = optim.Adam(LFEM1.parameters())
lfem2_optimizer = optim.Adam(LFEM2.parameters())
lfem3_optimizer = optim.Adam(LFEM3.parameters())
lfem4_optimizer = optim.Adam(LFEM4.parameters())
lfem5_optimizer = optim.Adam(LFEM5.parameters())
lfem6_optimizer = optim.Adam(LFEM6.parameters())

for epcoh in range(EPOCH):
    first_input_layer.train()
    LFEM1.train()
    LFEM2.train()
    LFEM3.train()
    LFEM4.train()
    LFEM5.train()
    LFEM6.train()

for epcoh in range(EPOCH):
    X_train, y_train = Variable(torch.tensor(X_train)), Variable(torch.tensor(y_train))
    X_train.to(device=device)
    y_train.to(device=device)

    first_optimizer.zero_grad()
    first_output = first_input_layer(X_train)
    first_loss = criterion(first_output, y_train)

    first_loss.backward()
    first_optimizer.step()
    print(first_loss)