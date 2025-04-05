import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from numba import cuda
#import pycuda.driver as cuda

def get_FEA_matrices(Fx_path, Fy_path, Fz_path):
    Fx_all = pd.read_excel(Fx_path, usecols=range(0,8),  nrows=6875,  header=None)
    Fy_all = pd.read_excel(Fy_path, usecols=range(0,8),  nrows=6875,  header=None)
    Fz_all = pd.read_excel(Fz_path, usecols=range(0,8),  nrows=6875,  header=None)
    # 将 DataFrame 转换为 NumPy 数组
    Fx_all_v = Fx_all.values
    # print(Fx_all_v.shape,Fx_all_v.ndim)
    Fy_all_v = Fy_all.values
    Fz_all_v = Fz_all.values

    # for index, row in enumerate(Fx_all_v[:, [0,2,4,6]]):
    #     #将离谱值替换为附近的值
    #     if np.any(row > -0.25):
    #         # print(f"index={index}, row={row}")
    #         Fx_all_v[index,[0,2,4,6]] = Fx_all_v[index-1,[0,2,4,6]]

    #选取需要的触点,Fz_R 取负值 F_FEA=[Fx1_R, Fx2_R, Fy_1_R, Fy2_R, Fz_1_R, Fz2_R, Fx1_L, Fx2_L, Fy_1_L, Fy2_L, Fz_1_L, Fz2_L]
    F_FEA = np.concatenate((Fx_all_v[:, [4, 6]], Fy_all_v[:, [5, 7]], Fz_all_v[:, [4, 6]], \
                            Fx_all_v[:, [0, 2]], Fy_all_v[:, [1, 3]], -Fz_all_v[:, [0, 2]]), axis=1)
    # print(F_FEA.shape)
    return F_FEA

class CustomDataset(Dataset):
    def __init__(self, data):
        # Assuming data is a NumPy array of shape (2000, 3, 14)
        print(data.shape)
        self.inputs = data[:, :, 6:]  # Last 8 floats are forces
        self.directions = data[:, :, 3:6]  # Directions are the 4th to 6th integers
        self.locations = data[:, :, :3]  # Locations are the first 3 integers

        # Normalize inputs
        #self.inputs = (self.inputs - self.inputs.mean()) / self.inputs.std()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.locations[idx], self.directions[idx]


def Normalize_F(F_matrix):
    for column in range(F_matrix.shape[1]):
        # Fx, Fy, Fz Min-Max 归一化至 [-1,1]
        F_matrix[:,column] = 2 * (F_matrix[:,column] - np.min(F_matrix[:,column])) / (np.max(F_matrix[:,column]) - np.min(F_matrix[:,column])) - 1

    F_normalized_matrix = F_matrix
    # print(f"normalized F shape= {F_normalized_matrix.shape}")
    return F_normalized_matrix

CHANGE_RATE=0.4
SAMPLE = 6000
num_samples = 5000
STEP=9

def change(change_rate = CHANGE_RATE):
    card = np.random.rand()
    if card<change_rate:
        return -1
    elif card>change_rate:
        return 1
    else:
        return 0

def tag_(disp):
    def tag(num):
        if num>2:
            return 1
        elif num<-2:
            return -1
        else:
            return 0
    disp.append(tag(disp[0]))
    disp.append(tag(disp[1]))
    disp.append(tag(disp[2]))

def tag_force (disp,FEA_DATA):
    rounded_list = FEA_DATA[round(275*disp[0]+11*disp[1]+disp[2]/3+3437)]
    rounded_list_without_y = rounded_list[:2].tolist()+ rounded_list[4:8].tolist()+ rounded_list[10:].tolist()
    disp = disp + rounded_list_without_y
    return disp

def diff_data_(arr):
    # Initialize the output array with zeros
    transformed_arr = np.zeros((arr.shape[0]-1, 17))
    
    # Calculate averages for the first 6 values
    for j in range(6):
        transformed_arr[:, j] = arr[1:, j]
    for j in range(6,9):
        transformed_arr[:,j] = arr[1:,j-6]-arr[:-1,j-6]
    # Calculate differences for the last 8 values
    for j in range(9, 17):
        transformed_arr[:, j] = arr[1:, j] - arr[:-1, j]
    return transformed_arr

def data_make(FEA_DATA):
    data = np.zeros([SAMPLE,STEP-1,17])
    sum = 0
    while sum<SAMPLE:
        jump = False
        steps = STEP
        disp=[[np.random.randint(-12,13),np.random.randint(-12,13),np.random.randint(-5,6)*3]]
        tag_(disp[0])
        disp[0]+=[0.,0.,0.]
        disp[0]=tag_force(disp[0],FEA_DATA)
        for index in range(steps-1):
            disp.append([disp[index][0]+change(),disp[index][1]+change(),disp[index][2]+change()*3])
            if (abs(disp[index+1][0])>12 or abs(disp[index+1][1])>12 or abs(disp[index+1][2])>15):
                jump = True
                continue
            tag_(disp[index+1])
            disp[index+1]+=[0.,0.,0.]
            disp[index+1]=tag_force(disp[index+1],FEA_DATA)
            #disp[index][0],disp[index][1],disp[index][2] = disp[index][0]/12,disp[index][1]/12,disp[index][2]/15
        if (jump == True):
            continue
        for _ in range(STEP):
            disp[_][0],disp[_][1],disp[_][2],disp[_][6],disp[_][7],disp[_][8] = disp[_][0]/12,disp[_][1]/12,disp[_][2]/15,disp[_][6]/12,disp[_][7]/12,disp[_][8]/15
        original_data=(np.array(disp,dtype=float))
        data[sum]= diff_data_(original_data)
        #print(data[sum])
        sum +=1
    #print(data.shape)
    return data
# Mock data for demonstration

#device = torch.device("cuda" )

F_FEA = get_FEA_matrices("./FEA_data/circle/PLA/1.30_Fx_circ.xlsx", \
                             "./FEA_data/circle/PLA/1.30_Fy_circ.xlsx", \
                             "./FEA_data/circle/PLA/1.30_Fz_circ.xlsx")
F_normalized_matrix = Normalize_F(F_FEA)
data = data_make(F_normalized_matrix)
dataset = CustomDataset(data)

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        #print(_x.dtype)
        x, _ = self.lstm(_x.float())  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


# Assuming 8 features for input, hidden size of 170, and 3 outputs for directions
direction_model = nn.Sequential(LSTMModel(11, 170, 108),nn.Linear(108,3))#.to(device)
location_model = nn.Sequential(LSTMModel(11, 170, 108),nn.Linear(108,3))#.to(device)

# Randomly select indices without replacement
selected_indices = np.random.choice(data.shape[0], num_samples, replace=False)
selected_data = data[selected_indices]
# Find the indices that were not selected
remaining_indices = np.setdiff1d(np.arange(data.shape[0]), selected_indices)

# Use the remaining indices to get the rest of the array
remaining_data = data[remaining_indices]
# Splitting the dataset into training and validation
#train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataset = CustomDataset(selected_data)
val_dataset = CustomDataset(remaining_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#@jit(nopython=True)
def train_model_direction(model, train_loader, val_loader, num_epochs=100):
    output = np.zeros([num_epochs,2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    time_start_1 = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, locations, directions in train_loader:
            #inputs, locations, directions = inputs.to(device), locations.to(device), directions.to(device)
            directions = directions.float()  # Ensure directions is a float tensor
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(inputs.dtype,inputs.shape)
            # print(outputs[0])  # Debug: Check output shape
            # print(directions[0])  # Debug: Check target shape
            loss = criterion(outputs, directions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, locations, directions in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, directions)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        output[epoch][0]=epoch+1
        output[epoch][1] = avg_val_loss
        print(f'Epoch {epoch+1}, Val Loss: {avg_val_loss}, Time cost:{time.time()-time_start_1}s, Time remain: {(time.time()-time_start_1)/(epoch+1)*(num_epochs-epoch)}')
    pd.DataFrame(output).to_csv("./file_direction.csv")

def train_model_location(model, train_loader, val_loader, num_epochs=100):
    output = np.zeros([num_epochs,2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    time_start_1 = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, locations, directions in train_loader:
            #inputs, locations, directions = inputs.to(device), locations.to(device), directions.to(device)
            locations = locations.float()  # Ensure directions is a float tensor
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(inputs.dtype,inputs.shape)
            # print(outputs[0])  # Debug: Check output shape
            # print(directions[0])  # Debug: Check target shape
            loss = criterion(outputs, locations)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, locations, directions in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, locations)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        output[epoch][0]=epoch+1
        output[epoch][1] = avg_val_loss
        print(f'Epoch {epoch+1}, Val Loss: {avg_val_loss}, Time cost:{time.time()-time_start_1}s, Time remain: {(time.time()-time_start_1)/(epoch+1)*(num_epochs-epoch)}')
    pd.DataFrame(output).to_csv("./file_location.csv")


# Training the direction model
# train_model_direction(direction_model, train_loader, val_loader, num_epochs=10000)
# torch.save(direction_model.state_dict(), './direction_model.pth')
train_model_location(location_model, train_loader, val_loader, num_epochs=10000)
torch.save(location_model.state_dict(), './location_model.pth')
