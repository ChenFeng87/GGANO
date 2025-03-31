import torch
import numpy as np
from torch import nn
import pickle
from sklearn.linear_model import LinearRegression


data_path_matrix = 'data10_matrix.pickle' 
with open(data_path_matrix, 'rb') as f:
    matrix = pickle.load(f)    


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
    
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=10, out_features=32, bias=True),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32, bias=True),
            nn.ReLU())        

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=10, bias=True)
            )

    def forward(self, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        output1 = torch.sigmoid(output)
        return output1

def init_node():
    initials=torch.rand(1,10)*0.1 + data_mean
    return initials

dynamics_learner = FullyConnected()

dynamics_learner.load_state_dict(torch.load('Parameters_saved.pickle'))
dynamics_learner.eval()

tt =np.linspace(0,1.0,100).reshape(-1, 1)
torch.manual_seed(2023)

data_mean = torch.ones([1,10])*0.5


k_num = 500
slope = torch.zeros([k_num,matrix.shape[1],matrix.shape[1]])

for num_t in range(k_num):
    print("num_t:",num_t)
    xx = init_node()
    for source in range(matrix.shape[1]):
        for target in range(matrix.shape[1]):
            if matrix[target,source]==1:
                f = torch.zeros([tt.shape[0]])
                output2_cut = xx.clone()
                regressor = LinearRegression()
                for jos in range(matrix.shape[1]):
                    if matrix[target,jos]==0:
                        output2_cut[:,jos] = 0
                for jot in range(f.shape[0]):
                    output3_cut = output2_cut.clone()
                    output3_cut[0,source] = tt[jot,0]
                    output4_cut = dynamics_learner(output3_cut)
                    f[jot] = output4_cut[0,target]
                regressor.fit(tt, f.detach().numpy())
                slope[num_t, source, target] = regressor.coef_[0]

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=np.inf)
torch.mean(slope, dim=0).numpy()

result_refer = torch.mean(slope, dim=0).numpy()
file_name = 'refer.npy'
np.save(file_name, result_refer)
