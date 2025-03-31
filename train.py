import torch
import numpy as np
from torch import nn
import pickle
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import time
import sys
# device = torch.device('cuda:7') 
# torch.cuda.set_device(device)


def load_data(batch_size=128):
    data_path = 'data.pickle'
    with open(data_path, 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)
    data_path_matrix = 'data10_matrix.pickle' 
    with open(data_path_matrix, 'rb') as f:
        matrix = pickle.load(f)     

    print('\n Train data size:', train_data.shape)
    print('\n Val data size:' , val_data.shape)
    print('\n Test data size:' , test_data.shape)
    print('\n AdjMatrix size:' , matrix.shape)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, matrix


def train_dynamics(args, dynamics_learner,
                   optimizer, device, train_loader, epoch, use_cuda, matrix_t):

    # dynamics_learner is the DNN model
    dynamics_learner.train()
    loss_records = []
    mse_records = []
    out_loss_record = []

    # train sub_epochs times before every validation
    for step in range(1, args.sub_epochs + 1):
        time3 = time.time()
        loss_record = []
        mse_record = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            loss, mse = train_dynamics_learner(optimizer, dynamics_learner, data, args.prediction_steps, use_cuda,
                                               device, matrix_t)
            loss_record.append(loss.item())
            mse_record.append(mse.item())
        loss_records.append(np.mean(loss_record))
        mse_records.append(np.mean(mse_record))
        print('\nTraining %d/%d before validation, loss: %f, MSE: %f' % (step,args.sub_epochs, np.mean(loss_record), np.mean(mse_record)))
        time4 = time.time()
        print("it spends %d seconds for trainsing this sub-epoch" % (time4-time3))

        out_loss_record.append(np.mean(loss_record))
    return out_loss_record


def val_dynamics(args, dynamics_learner, device, val_loader, epoch, best_val_loss, use_cuda, matrix_t):

    dynamics_learner.eval()
    loss_record = []
    mse_record = []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        loss, mse = val_dynamics_learner(dynamics_learner, data, args.prediction_steps, use_cuda, device, matrix_t)
        loss_record.append(loss.item())
        mse_record.append(mse.item())

    print('\nValidation: loss: %f, MSE: %f' % (np.mean(loss_record), np.mean(mse_record)))

    if best_val_loss > np.mean(loss_record):
        torch.save(dynamics_learner.state_dict(), args.dynamics_path)

    return np.mean(loss_record)


def train_dynamics_learner(optimizer, dynamics_learner, data, steps, use_cuda, device, matrix_t):
    optimizer.zero_grad()

    input1 = data[:, :, 0, :]
    target = data[:, :, 1: steps, :]
    output = input1

    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
    outputs = outputs.cuda() if use_cuda else outputs
    # Make a prediction with steps-1，output：batchsize, num_nodes, time_steps, dimension
    for t in range(steps - 1):
        output1 = torch.as_tensor(output, dtype=torch.float32).view(-1, data.size(1)).to(device)
        output3 = dynamics_learner(output1)

        output_11 = output3.clone() 
        for ios in range(output_11.shape[1]):
            output2_cut = output1.clone()
            for jos in range(output_11.shape[1]):
                if matrix_t[ios,jos]==0:
                    output2_cut[:,jos] = 0
            output3_cut = dynamics_learner(output2_cut)
            output_11[:,ios] = output3_cut[:,ios].clone()

        output = output1 + output_11*0.01 - output1 * 0.01 * 1
        out11 = output.view(-1, data.size(1), 1)
        outputs[:, :, t, :] = out11

    loss = torch.mean(torch.abs(outputs - target))
    loss.backward()
    optimizer.step()
    mse = F.mse_loss(outputs, target)
    if use_cuda:
        loss = loss.cpu()
        mse = mse.cpu()

    return loss, mse


def val_dynamics_learner(dynamics_learner, data, steps, use_cuda, device, matrix_t):

    input1 = data[:, :, 0, :]
    target = data[:, :, 1: steps, :]
    output = input1

    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
    outputs = outputs.cuda() if use_cuda else outputs
    for t in range(steps - 1):
        output1 = torch.as_tensor(output, dtype=torch.float32).view(-1, data.size(1)).to(device)
        output3 = dynamics_learner(output1)

        output_11 = output3.clone()  # 存放cut后的f1~f10
        # print("output_11:",output_11[0:3])
        for ios in range(output_11.shape[1]):
            output2_cut = output1.clone()
            # print("output_2:",output2_cut[0:3])
            for jos in range(output_11.shape[1]):
                if matrix_t[ios,jos]==0:
                    output2_cut[:,jos] = 0
            # print("output2_cut:",output2_cut[0:3])
            output3_cut = dynamics_learner(output2_cut)
            # print("shape:",output3_cut.shape)
            # print("output_cut:",output3_cut[0:3])
            output_11[:,ios] = output3_cut[:,ios].clone()
            # print("output_11_new:",output_11[0:3])
        # sys.exit()

        output = output1 + output_11 * 0.01 - output1 * 0.01 * 1
        out11 = output.view(-1, data.size(1), 1)
        outputs[:, :, t, :] = out11

    loss = torch.mean(torch.abs(outputs - target))
    mse = F.mse_loss(outputs, target)

    return loss, mse


def test(args, dynamics_learner, device, test_loader, use_cuda, matrix_t):
    # load model
    dynamics_learner.load_state_dict(torch.load(args.dynamics_path))
    dynamics_learner.eval()
    loss_record = []
    mse_record = []
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        loss, mse = val_dynamics_learner(dynamics_learner, data, args.prediction_steps, use_cuda, device, matrix_t)
        loss_record.append(loss.item())
        mse_record.append(mse.item())
    print('loss: %f, mse: %f' % (np.mean(loss_record), np.mean(mse_record)))


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


def main():

    # Training settings

    parser = argparse.ArgumentParser(description='DNN_FOR_NETWORK_REFERENCE')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1998,
                        help='random seed (default: 1998)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs, default: 5)')
    parser.add_argument('--sub-epochs', type=int, default=1,
                        help='i.e. train 10 times before every Validation (default: 10)')
    parser.add_argument('--prediction-steps', type=int, default=20,
                        help='prediction steps in data (default: 200)')
    parser.add_argument('--dynamics-path', type=str, default='Parameters_saved.pickle',
                        help='path to save dynamics learner (default: ./Parameters_saved.pickle)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    torch.manual_seed(args.seed)

    # Loading data
    print('\n----------   Loading data ----------')
    train_loader, val_loader, test_loader, adj_matrix = load_data(batch_size=128)
    print('\n----------   loading data is finished ----------')

    # move network to gpu
    dynamics_learner = FullyConnected().to(device)
    # Adam optimizer and the learning rate is 1e-3
    optimizer = torch.optim.Adam(dynamics_learner.parameters(), lr=0.001)

    # Initialize the best validation error and corresponding epoch
    best_val_loss = np.inf
    best_epoch = 0

    loss_out = []
    print('\n----------   Parameters of each layer  ----------')
    for name, parameters in dynamics_learner.named_parameters():
        print(name,":",parameters.shape)

    print('\n----------   begin training  ----------')
    print('\n--   You need to wait about 10 minutes for each sub-epoch ')
    for epoch in range(1, args.epochs + 1):
        print(device)
        time1 = time.time()
        print('\n----------   Epoch %d/%d ----------' % (epoch,args.epochs))
        out_loss = train_dynamics(args, dynamics_learner, optimizer, device, train_loader,
                                  epoch, use_cuda, adj_matrix)
        val_loss = val_dynamics(args, dynamics_learner, device, val_loader,
                                epoch, best_val_loss, use_cuda, adj_matrix)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        print('\nCurrent best epoch: %d, best val loss: %f' % (best_epoch, best_val_loss))

        loss_out.append(out_loss)
        time2 =time.time()
        print("it spends %d seconds for trainsing this epoch" % (time2-time1))


    print('\nBest epoch: %d' % best_epoch)

    test(args, dynamics_learner, device, test_loader, use_cuda, adj_matrix)

    # # output the loss
    # loss_address = 'loss.pickle'
    # with open(loss_address, 'wb') as f:
    #     pickle.dump(loss_out, f)

    print('\n-----The code finishes running' )

if __name__ == '__main__':
    main()