import numpy as np
import torch.autograd
import os
import time
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class Network(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        with_batch_norm=False,
    ):
        super(Network, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_size, hidden_sizes[0]))
        if with_batch_norm:
            self.layers.append(nn.LayerNorm(normalized_shape=(hidden_sizes[0])))
        self.layers.append(nn.ReLU())
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if with_batch_norm:
                self.layers.append(nn.LayerNorm(normalized_shape=(hidden_sizes[i+1])))
            self.layers.append(nn.ReLU())
        
        self.actionsLayer = nn.Linear(hidden_sizes[-1], self.output_size)
        
        self.sigmoidLayer = nn.Linear(hidden_sizes[-1], 1)
        
        
    def forward(self, x):
        out = x
        
        for layer in self.layers:
            out = layer(out)
        
        actions = torch.tanh(self.actionsLayer(out))
        sigmoid = torch.sigmoid(self.sigmoidLayer(out))
        
        return actions, sigmoid


def save_weights_and_graph(save_dir):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), save_dir + 'model.pt')

    np.save(save_dir + '/trainLosses.npy', np.stack(trainLosses))
    np.save(save_dir +'/testLosses.npy', np.stack(testLosses))

idx = 0
save_dir = 'models/MLP-inverseDynamics/with-sigmoid/' + str(idx) + '/'

prefix = 'datasets/' + str(idx) + '/'
states = np.load(prefix + 'states_array.npy')
actions = np.load(prefix + 'actions_array.npy')
rewards = np.load(prefix + 'rewards_array.npy')
next_states = np.load(prefix + 'next_states_array.npy')
dones = np.load(prefix + 'dones_array.npy')

X = np.concatenate((states, next_states), -1)
Y = actions

permutation = np.random.permutation(X.shape[0])
X = X[permutation]
X_test = torch.from_numpy(X[:100000]).float().to(device)
X_train = torch.from_numpy(X[100000:]).float().to(device)
Y = Y[permutation]
Y_test = torch.from_numpy(Y[:100000]).float().to(device)
Y_train = torch.from_numpy(Y[100000:]).float().to(device)

lr = 1e-5
batch_size = 1024
numBatches = int(np.ceil(X_train.shape[0] / batch_size))

model = Network(X_train.shape[1], Y_train.shape[1], hidden_sizes=[256, 256], with_batch_norm=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
lmbda = lambda epoch: 0.8
lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lmbda)
l2Loss  = nn.MSELoss(reduction='none')
binaryLoss = nn.BCELoss()
zeroTensor = torch.zeros([batch_size, 1]).to(device)
oneTensor = torch.ones([batch_size, 1]).to(device)
zeroTensorTest = torch.zeros([X_test.shape[0], 1]).to(device)
oneTensorTest = torch.ones([X_test.shape[0], 1]).to(device)
binaryLossWeighing = 1e-8

trainLosses = []
testLosses = []
learningRates = []

for epoch in range(200):
    
    with torch.no_grad():
        
        testLosses.append(np.zeros(5))

        y_actions, y_sigmoid = model.forward(X_test)
        
        actionsLoss = l2Loss(y_actions, Y_test)
        sigmoidLoss = binaryLossWeighing * binaryLoss(y_sigmoid, oneTensorTest)
        
        testLosses[-1][0] = actionsLoss.mean().item()
        testLosses[-1][1] = sigmoidLoss.item()
        testLosses[-1][2] = torch.eq(oneTensorTest, torch.round(y_sigmoid)).sum() / float(X_test.shape[0])
        
        totalForwardLoss = actionsLoss.mean() + sigmoidLoss
        
        xHalfLength = int(X_test.shape[-1] / 2)
        y_actions, y_sigmoid = model.forward(torch.cat((X_test[:, xHalfLength:], X_test[:, :xHalfLength]), -1))
        backwardLoss = binaryLossWeighing * binaryLoss(y_sigmoid, zeroTensorTest)

        testLosses[-1][3] = backwardLoss.item()
        testLosses[-1][4] = torch.eq(zeroTensorTest, torch.round(y_sigmoid)).sum() / float(X_test.shape[0])
        
    print('Test Loss', np.round(testLosses[-1], decimals=3))
                
    t0 = time.time()

    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    Y_train = Y_train[permutation]
    
    epochLoss = 0
    for batch in range(numBatches-1):
        
        optimizer.zero_grad()
        trainLosses.append(np.zeros(5))
        
        x = X_train[batch * batch_size:(batch+1)*batch_size]
        y = Y_train[batch * batch_size:(batch+1)*batch_size]
        
        y_actions, y_sigmoid = model.forward(x)
        
        actionsLoss = l2Loss(y_actions, y)
        sigmoidLoss = binaryLossWeighing * binaryLoss(y_sigmoid, oneTensor)
        
        trainLosses[-1][0] = actionsLoss.mean().item()
        trainLosses[-1][1] = sigmoidLoss.item()
        trainLosses[-1][2] = torch.eq(oneTensor, torch.round(y_sigmoid)).sum() / float(batch_size)
        
        totalForwardLoss = actionsLoss.mean() + sigmoidLoss
        totalForwardLoss.backward()
        
        xHalfLength = int(x.shape[-1] / 2)
        y_actions, y_sigmoid = model.forward(torch.cat((x[:, xHalfLength:], x[:, :xHalfLength]), -1))
        backwardLoss = binaryLossWeighing * binaryLoss(y_sigmoid, zeroTensor)
        backwardLoss.backward()
        
        trainLosses[-1][3] = backwardLoss.item()
        trainLosses[-1][4] = torch.eq(zeroTensor, torch.round(y_sigmoid)).sum() / float(batch_size)
        
        optimizer.step()
        
    save_weights_and_graph(save_dir)
    print('Epoch Loss', np.round(trainLosses[-1], decimals=3))

