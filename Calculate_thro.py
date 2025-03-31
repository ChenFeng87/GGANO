
import scipy.io as sio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle

def plot_graph(adjacency_matrix, title, pos=None, ax=None, k=0.5):
    G = nx.Graph()
    num_nodes = len(adjacency_matrix)
    
    for i in range(num_nodes):
        G.add_node(i)
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)

    if pos is None:
        pos = nx.spring_layout(G, k=k)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
    ax.set_title(title)
    
    return pos

def log_Likelihood(tho, S_cov, Theta, n):
    matrix = Theta.copy()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if np.abs(matrix[i,j])<=tho:
                matrix[i,j]=0

    if matrix.shape[0] != matrix.shape[1]:  
        raise ValueError("Theta must be a square matrix")  
      
    if S_cov.shape[1] != matrix.shape[0]:  
        raise ValueError("The dimensions of S and Theta are inconsistent") 
    
    log_det = np.linalg.slogdet(matrix)[1]  # slogdet return (sign, log|det|)  
      
    # calculate tr(S Theta)  
    tr_SA = np.trace(np.dot(S_cov, matrix))  
      
    # Calculate the final result
    log_det_minus_tr = log_det - tr_SA  
      
    return log_det_minus_tr * n

def percentage(tho, Theta):
    matrix = np.abs(Theta)
    matrix[matrix > tho] = 1  
    matrix[matrix <= tho] = 0 
    sum_m = np.sum(matrix)
    return (sum_m-10)/2/10/9*100, matrix


np.set_printoptions(precision=4)
np.set_printoptions(linewidth=4000)

cell_S_path = 'Emp_cov.mat' 
all_S = sio.loadmat(cell_S_path)
data_S = all_S['S_i']

tgfb1_S = data_S[:,0]

cell_matrix_path = 'pprediction_network.mat' 
all_matrix = sio.loadmat(cell_matrix_path)
data_matrix = all_matrix['Theta0']

tgfb_matrix = data_matrix[:,0]

TT = 20

matrix_sum_tgfb = np.zeros([10,10])
for os in range(tgfb_matrix.shape[0]):
    matrix_sum_tgfb = matrix_sum_tgfb + np.abs(tgfb_matrix[os])
matrix_sum_tgfb = matrix_sum_tgfb/TT

matrix_sum_S_i = np.zeros([10,10])
for os in range(tgfb_matrix.shape[0]):
    matrix_sum_S_i = matrix_sum_S_i + tgfb1_S[os]
matrix_sum_S_i = matrix_sum_S_i/TT


t_space = np.linspace(0,0.1, 40)
Likelihood_value = np.zeros([1, t_space.shape[0]])
matrix_list = []
sample_N_1 = [1]*TT


for j in range(t_space.shape[0]):
    Likelihood_value[0, j] = Likelihood_value[0, j] + log_Likelihood(t_space[j], matrix_sum_S_i, matrix_sum_tgfb, 1)
        
matrix_list = []

index_of_max = np.argmax(Likelihood_value[0,:]) 
number_sum, matrix_res = percentage(t_space[index_of_max], matrix_sum_tgfb)
matrix_list.append(matrix_res)
print(f"tgfb1: the maximum value is: {index_of_max} and the thre: {t_space[index_of_max]} and the percentage:{number_sum} %")


##
num_times = len(matrix_list)
pos = None
pos = plot_graph(np.array(matrix_list[0]).astype(int), '10 dim', pos=pos, k = 0.7)

plt.tight_layout()
plt.savefig('Undirected Graph.jpg', dpi=600)
plt.show()

matrix = torch.zeros(10, 10)
matrix.fill_diagonal_(1)
for i in range(9):
    matrix[i, i + 1] = 1
    matrix[i + 1, i] = 1
matrix[0,9] = 1
matrix[9,0] = 1
print(matrix)
with open('data10_matrix.pickle', 'wb') as f:
    pickle.dump(matrix, f)

##
indices = np.indices(matrix_sum_tgfb.shape)  
non_diagonal_elements = matrix_sum_tgfb[indices[0] != indices[1]]  
bins = np.linspace(0, 0.1, 100)  
plt.figure(figsize=(5, 3.5))
plt.hist(non_diagonal_elements, bins=bins, edgecolor='black', alpha=0.7, histtype='bar')  
plt.axvline(x=0.03, color='red', linestyle='--', linewidth=1)  
plt.grid(axis='y', alpha=0.75)  
plt.savefig('hist.png', dpi=600)
plt.show()

##
matrix = np.random.rand(10, 10)  
np.fill_diagonal(matrix, np.nan)

plt.figure(figsize=(6, 4))
annot_matrix = np.floor(matrix_sum_tgfb * 100) / 100  
sns.heatmap(
    matrix_sum_tgfb,
    annot=annot_matrix,
    fmt=".2f",
    cmap="viridis",
    mask=np.isnan(matrix),
    cbar=True,
    vmin=0, 
    vmax=0.12,  
    linewidths=0.5,
    linecolor='white'
)
plt.savefig('heatmap.png', dpi=600)
plt.show()

##
plt.figure(figsize=(5, 3.5))
plt.plot(t_space, Likelihood_value[0,:], color='royalblue', linestyle='-', marker='s', markersize=8, 
         markerfacecolor='white', markeredgewidth=2, markeredgecolor='darkblue', 
         linewidth=2, label='Sample Line')
plt.axvline(x=0.03, color='red', linestyle='--', linewidth=1)   
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('likehood.png', dpi=600)
plt.show()


##
indices = np.indices(matrix_sum_tgfb.shape)
non_diagonal_elements = matrix_sum_tgfb[indices[0] != indices[1]]
bins = np.linspace(0, 0.1, 100)

data_leq_003 = non_diagonal_elements[non_diagonal_elements <= 0.03]
data_gt_003 = non_diagonal_elements[non_diagonal_elements > 0.03]

plt.figure(figsize=(5, 3.5))
plt.hist(data_leq_003, bins=bins, edgecolor='black', alpha=0.7, histtype='bar', label='.............')
plt.hist(data_gt_003, bins=bins, edgecolor='black', alpha=0.7, histtype='bar', color='orange', label='""""'"""""""")
plt.axvline(x=0.03, color='red', linestyle='--', linewidth=1)

plt.grid(axis='y', alpha=0.75)
plt.savefig('hist_color_fign.png', dpi=600)
plt.show()
