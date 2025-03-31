import numpy as np
from scipy import io
import torch
import pickle

def move(Point, steps, sets):
    a, b, s, n, k = sets
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = Point

    dx1 = a*x1**n / (s**n + x1**n) + b*s**n/(x10**n+s**n)-k*x1
    dx2 = a*x2**n / (s**n + x2**n) + b*s**n/(x1**n+s**n)-k*x2
    dx3 = a*x3**n / (s**n + x3**n) + b*s**n/(x2**n+s**n)-k*x3
    dx4 = a*x4**n / (s**n + x4**n) + b*s**n/(x3**n+s**n)-k*x4
    
    dx5 = a*x5**n / (s**n + x5**n) + b*s**n/(x4**n+s**n)-k*x5
    dx6 = a*x6**n / (s**n + x6**n) + b*s**n/(x5**n+s**n)-k*x6
    dx7 = a*x7**n / (s**n + x7**n) + b*s**n/(x6**n+s**n)-k*x7
    dx8 = a*x8**n / (s**n + x8**n) + b*s**n/(x7**n+s**n)-k*x8
    dx9 = a*x9**n / (s**n + x9**n) + b*s**n/(x8**n+s**n)-k*x9
    dx10 = a*x10**n / (s**n + x10**n) + b*s**n/(x9**n+s**n)-k*x10
    
    return [x1 + dx1 * steps+ np.random.normal(loc=0, scale=siga, size=1) , x2 + dx2 * steps+ np.random.normal(loc=0, scale=siga, size=1), 
            x3 + dx3 * steps+ np.random.normal(loc=0, scale=siga, size=1) , x4 + dx4 * steps+ np.random.normal(loc=0, scale=siga, size=1),
            x5 + dx5 * steps+ np.random.normal(loc=0, scale=siga, size=1) , x6 + dx6 * steps+ np.random.normal(loc=0, scale=siga, size=1),
            x7 + dx7 * steps+ np.random.normal(loc=0, scale=siga, size=1) , x8 + dx8 * steps+ np.random.normal(loc=0, scale=siga, size=1),
            x9 + dx9 * steps+ np.random.normal(loc=0, scale=siga, size=1) , x10 + dx10 * steps+ np.random.normal(loc=0, scale=siga, size=1)]

def init_node():
    initials=np.random.random(10)
    return initials

print("***Start generating data***")
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
x9 = []
x10 = []
np.random.seed(2022)
NN=20
tt = 0.01
a=0.5
b=0.8
DD = 0.30
siga = np.sqrt(2*DD*tt)
print("sigma:", siga)
for i in range(20000):
    print("number:", i)
    P0 = init_node()
    P = P0
    xx1 = np.zeros([NN])
    xx2 = np.zeros([NN])
    xx3 = np.zeros([NN])
    xx4 = np.zeros([NN])
    xx5 = np.zeros([NN])
    xx6 = np.zeros([NN])
    xx7 = np.zeros([NN])
    xx8 = np.zeros([NN])
    xx9 = np.zeros([NN])
    xx10 = np.zeros([NN])
    for v in range(NN):
        P = move(P, tt, np.array([a, b, 0.2, 4.0, 1.0]))
        for j in range(10):
            if P[j]<0:
                P[j]=0
        xx1[v]=P[0]
        xx2[v]=P[1]
        xx3[v]=P[2]
        xx4[v]=P[3]
        xx5[v]=P[4]
        xx6[v]=P[5]
        xx7[v]=P[6]
        xx8[v]=P[7]
        xx9[v]=P[8]
        xx10[v]=P[9]
    
    x1.append(xx1)
    x2.append(xx2)
    x3.append(xx3)
    x4.append(xx4)
    x5.append(xx5)
    x6.append(xx6)
    x7.append(xx7)
    x8.append(xx8)
    x9.append(xx9)
    x10.append(xx10)

datax1=np.array(x1)
datax2=np.array(x2)
datax3=np.array(x3)
datax4=np.array(x4)
datax5=np.array(x5)
datax6=np.array(x6)
datax7=np.array(x7)
datax8=np.array(x8)
datax9=np.array(x9)
datax10=np.array(x10)

m, n = datax1.shape
youtcome = np.zeros([m, 10, n, 1])

for i in range(m):
    youtcome[i, 0, :, :] = datax1[i, :].reshape(n, 1)
    youtcome[i, 1, :, :] = datax2[i, :].reshape(n, 1)
    youtcome[i, 2, :, :] = datax3[i, :].reshape(n, 1)
    youtcome[i, 3, :, :] = datax4[i, :].reshape(n, 1)
    youtcome[i, 4, :, :] = datax5[i, :].reshape(n, 1)
    youtcome[i, 5, :, :] = datax6[i, :].reshape(n, 1)
    youtcome[i, 6, :, :] = datax7[i, :].reshape(n, 1)
    youtcome[i, 7, :, :] = datax8[i, :].reshape(n, 1)
    youtcome[i, 8, :, :] = datax9[i, :].reshape(n, 1)
    youtcome[i, 9, :, :] = datax10[i, :].reshape(n, 1)

new_data = np.zeros([400000,10])
for i in range(20000):
    for j in range(20):
        new_data[i*20+j,:] = youtcome[i,:,j,0]

io.savemat('Data.mat', {'array': new_data})


index = [o for o in range(youtcome.shape[0])]
np.random.shuffle(index)
ynew = youtcome[index, :, :, :]
train_data = ynew[: youtcome.shape[0] // 10 * 8, :, :, :]
val_data = ynew[youtcome.shape[0] // 10 * 8:youtcome.shape[0] // 10 * 9, :, :, :]
test_data = ynew[youtcome.shape[0] // 10 * 9:, :, :, :]

train_data = torch.tensor(train_data)
val_data = torch.tensor(val_data)
test_data = torch.tensor(test_data)
print('\n Train data size:', train_data.shape)
print('\n Val data size:' , val_data.shape)
print('\n Test data size:' , test_data.shape)

print('\n----------   Finsh generating time series data ----------')
results = [train_data, val_data, test_data]
with open('data.pickle', 'wb') as f:
    pickle.dump(results, f)

print("***Data is ready***")