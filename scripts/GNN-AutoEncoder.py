#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import torch.autograd
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
import dgl
from graphenvs import HalfCheetahGraphEnv
import itertools
import os

class Network(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        with_batch_norm=False,
        activation=None
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
        
        self.layers.append(nn.Linear(hidden_sizes[len(hidden_sizes) - 1], self.output_size))
        
        if activation is not None:
            self.layers.append(activation())
            
    def forward(self, x):
        out = x
        
        for layer in self.layers:
            out = layer(out)
            
        return out

class GraphNeuralNetwork(nn.Module):
    def __init__(
        self,
        inputNetwork,
        messageNetwork,
        updateNetwork,
        outputNetwork,
        numMessagePassingIterations,
        encoder = True
    ):
        
        super(GraphNeuralNetwork, self).__init__()
                
        self.inputNetwork = inputNetwork
        self.messageNetwork = messageNetwork
        self.updateNetwork = updateNetwork
        self.outputNetwork = outputNetwork
        
        self.numMessagePassingIterations = numMessagePassingIterations
        self.encoder = encoder
        
    def inputFunction(self, nodes):
        return {'state' : self.inputNetwork(nodes.data['input'])}
    
    def messageFunction(self, edges):
        
        batchSize = edges.src['state'].shape[1]
        edgeData = edges.data['feature'].repeat(batchSize, 1).T.unsqueeze(-1)
        nodeInput = edges.src['input']
        
#         print(edges.src['state'].shape)
#         print(nodeInput.shape)
        return {'m' : self.messageNetwork(torch.cat((edges.src['state'], edgeData, nodeInput), -1))}
        

    def updateFunction(self, nodes):
        return {'state': self.updateNetwork(torch.cat((nodes.data['m_hat'], nodes.data['state']), -1))}
    
    def outputFunction(self, nodes):
        
#         numNodes, batchSize, stateSize = graph.ndata['state'].shape
#         return self.outputNetwork.forward(graph.ndata['state'])
        return {'output': self.outputNetwork(nodes.data['state'])}


    def forward(self, graph, state):
        
        self.update_states_in_graph(graph, state)
        
        graph.apply_nodes(self.inputFunction)
        
        for messagePassingIteration in range(self.numMessagePassingIterations):
            graph.update_all(self.messageFunction, dgl.function.max('m', 'm_hat'), self.updateFunction)
        
        graph.apply_nodes(self.outputFunction)
        
        output = graph.ndata['output']

        if self.encoder:
            output = F.normalize(output, dim=-1)

        return output
    
    def update_states_in_graph(self, graph, state):
        
        if self.encoder:
            if len(state.shape) == 1:
                state = state.unsqueeze(0)

            numGraphFeature = 6
            numGlobalStateInformation = 5
            numLocalStateInformation = 2
            numStateVar = state.shape[1]
            globalInformation = state[:, 0:5]
            batch_size = state.shape[0]
            numNodes = (numStateVar - 5) // 2

            nodeData = torch.empty((numNodes, batch_size, numGraphFeature + numGlobalStateInformation + numLocalStateInformation)).to(device)

            nodeData[:, :, 0:numGlobalStateInformation] = globalInformation            
            for nodeIdx in range(numNodes):
                # Assign local state information
                nodeData[nodeIdx, :, numGlobalStateInformation] = state[:, 5 + nodeIdx]
                nodeData[nodeIdx, :, numGlobalStateInformation + 1] = state[:, 5 + numNodes + nodeIdx]
                # Assign global features from graph
                nodeData[nodeIdx, :, numGlobalStateInformation + 2 : numGlobalStateInformation + 2 + numGraphFeature] = graph.ndata['feature'][nodeIdx]

            graph.ndata['input'] = nodeData
        
        else:
            numNodes, batchSize, inputSize = state.shape
            nodeData = torch.empty((numNodes, batchSize, inputSize + 6)).to(device)
            nodeData[:, :, :inputSize] = state
            nodeData[:, :, inputSize : inputSize + 6] = graph.ndata['feature'].unsqueeze(dim=1).repeat_interleave(batchSize, dim=1)
#             for nodeIdx in range(numNodes):
#                 nodeData[nodeIdx, :, inputSize : inputSize + 6] = graph.ndata['feature'][nodeIdx]
            
            graph.ndata['input'] = nodeData
        

states = {}
actions = {}
rewards = {}
next_states = {}
dones = {}
env = {}

# idx = 5
# trainingIdxs = [idx]

trainingIdxs = [0, 1, 2, 3, 4, 5]


# save_dir = 'models/single-GNN-2-latent-no-contrastive/' + str(idx) + /
save_dir = 'models/multi-GNN-4-latent-contrastive/'


def save_weights_and_graph(save_dir):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.save(encoderGNN.state_dict(), save_dir + 'encoderGNN.pt')
    torch.save(decoderGNN.state_dict(), save_dir + 'decoderGNN.pt')

    for morphIdx in trainingIdxs:
        np.save(save_dir + str(morphIdx) + '-testLosses', np.stack(testLosses[morphIdx]))
        np.save(save_dir + str(morphIdx) + '-trainLosses', np.stack(trainLosses[morphIdx]))

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Average L2 Loss')
    plt.title('Testing Set Reconstruction Loss Per Morphology')
    if len(trainingIdxs) > 1:
        plt.legend(trainingIdxs)
    plt.savefig('time-contrastive-losses.jpg')


for morphIdx in trainingIdxs:

    prefix = 'datasets/{}/'.format(morphIdx)
    
    states[morphIdx] = np.load(prefix + 'states_array.npy')
    actions[morphIdx] = np.load(prefix + 'actions_array.npy')
    rewards[morphIdx] = np.load(prefix + 'rewards_array.npy')
    next_states[morphIdx] = np.load(prefix + 'next_states_array.npy')
    dones[morphIdx] = np.load(prefix + 'dones_array.npy')
    
    env[morphIdx] = HalfCheetahGraphEnv(None)
    env[morphIdx].set_morphology(morphIdx)


# In[5]:


X_test = {}
X_train = {}
Y_test = {}
Y_train = {}

for morphIdx in trainingIdxs:
    X = states[morphIdx]
    Y = next_states[morphIdx]
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    X_test[morphIdx] = torch.from_numpy(X[:100000]).float()
    X_train[morphIdx] = torch.from_numpy(X[100000:]).float()
    Y = Y[permutation]
    Y_test[morphIdx] = torch.from_numpy(Y[:100000]).float()
    Y_train[morphIdx] = torch.from_numpy(Y[100000:]).float()


# In[12]:


hidden_sizes = [256, 256]

inputSize = 13
stateSize = 64
messageSize = 64
latentSize = 4
numMessagePassingIterations = 6
batch_size = 1024
numBatchesPerTrainingStep = 1
minDistanceSeqAndRand = 0.25
with_batch_norm = True

# # Encoder Networks 
encoderInputNetwork = Network(inputSize, stateSize, hidden_sizes, with_batch_norm=with_batch_norm)
encoderMessageNetwork = Network(stateSize + inputSize + 1, messageSize, hidden_sizes, with_batch_norm=with_batch_norm, activation=nn.Tanh)
encoderUpdateNetwork = Network(stateSize + messageSize, stateSize, hidden_sizes, with_batch_norm=with_batch_norm)
encoderOutputNetwork = Network(stateSize, latentSize, hidden_sizes, with_batch_norm=with_batch_norm)
encoderGNN = GraphNeuralNetwork(encoderInputNetwork, encoderMessageNetwork, encoderUpdateNetwork, encoderOutputNetwork, numMessagePassingIterations, encoder=True).to(device)

# # Decoder Networks
decoderInputNetwork = Network(latentSize + 6, stateSize, hidden_sizes, with_batch_norm=with_batch_norm)
decoderMessageNetwork = Network(stateSize + latentSize + 7, messageSize, hidden_sizes, with_batch_norm=with_batch_norm, activation=nn.Tanh)
decoderUpdateNetwork = Network(stateSize + messageSize, stateSize, hidden_sizes, with_batch_norm=with_batch_norm)
decoderOutputNetwork = Network(stateSize, 7, hidden_sizes, with_batch_norm=with_batch_norm)
decoderGNN = GraphNeuralNetwork(decoderInputNetwork, decoderMessageNetwork, decoderUpdateNetwork, decoderOutputNetwork, numMessagePassingIterations, encoder=False).to(device)

# Optimizer
lr = 5e-4
optimizer = optim.Adam(itertools.chain(
                    encoderInputNetwork.parameters(), encoderMessageNetwork.parameters(), 
                    encoderUpdateNetwork.parameters(), encoderOutputNetwork.parameters(),
                    decoderInputNetwork.parameters(), decoderMessageNetwork.parameters(), 
                    decoderUpdateNetwork.parameters(), decoderOutputNetwork.parameters()),
                    lr, weight_decay=0)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, verbose=True, min_lr=5e-6, threshold=1e-2)
criterion = nn.MSELoss(reduction='none')


numTrainingBatches = int(np.ceil(X_train[trainingIdxs[0]].shape[0] / batch_size))
numTestingBatches = int(np.ceil(X_test[trainingIdxs[0]].shape[0] / batch_size))

zeroTensor = torch.zeros([1]).to(device)
trainLosses = {}
testLosses = {}
validLosses = {}

for morphIdx in trainingIdxs:
    trainLosses[morphIdx] = []
    testLosses[morphIdx] = []
    validLosses[morphIdx] = []


for epoch in range(20):
    
    print('Starting Epoch {}'.format(epoch), flush=True)
    epoch_t0 = time.time()
    
    for morphIdx in trainingIdxs:
        permutation = np.random.permutation(X_train[morphIdx].shape[0])
        X_train[morphIdx] = X_train[morphIdx][permutation]
        Y_train[morphIdx] = Y_train[morphIdx][permutation]

    s = 0
    with torch.no_grad():
        for morphIdx in trainingIdxs:
            testLosses[morphIdx].append(np.zeros(2))
            for batch_ in range(0, numTestingBatches-1):

                encoder_graph = env[morphIdx].get_graph()._get_dgl_graph()
                decoder_graph = env[morphIdx].get_graph()._get_dgl_graph()

                current_states = X_test[morphIdx][batch_ * batch_size:(batch_+1)*batch_size]
                next_states = Y_test[morphIdx][batch_ * batch_size:(batch_+1)*batch_size]
                random_indexes = np.random.choice(X_test[trainingIdxs[0]].shape[0], size=batch_size, replace=False)
                random_states = X_test[morphIdx][random_indexes]

                encoderInput = torch.cat((current_states, next_states, random_states), dim=0).to(device)

                latent_states = encoderGNN(encoder_graph, encoderInput)
                current_state_reconstruction = decoderGNN(decoder_graph, latent_states[:, 0:current_states.shape[0], :])
                current_state_reconstruction[:, :, 0:5] = current_state_reconstruction[:, :, 0:5].mean(dim=0)

                autoencoder_loss = criterion(encoder_graph.ndata['input'][:, 0:batch_size, :7], current_state_reconstruction).mean()
                # Calculate 2-norm for positive/sequential samples over the data dimension - result is of dimension (nodes, batch_size)
                sequential_distances = torch.norm(latent_states[:, 0:batch_size, :] - latent_states[:, batch_size:batch_size * 2, :], p=None, dim=2)
                # Calculate 2-norm for negative/random samples over data dimension - result is of dimension (nodes, batch_size)
                random_distances = torch.norm(latent_states[:, 0:batch_size, :] - latent_states[:, 2 * batch_size: 3 * batch_size, :], p=None, dim=2)
                # Calculate contrastive loss for each entry - result is of dimension (nodes, batch_size)
                contrastive_loss = torch.max(zeroTensor, sequential_distances - random_distances + minDistanceSeqAndRand)
                # get 0-1 matrix which is True if entry is not 0
                mask = contrastive_loss != 0
                # Compute average over nonzero entries in batch, result will be scalar
                final_contrastive_loss = contrastive_loss.sum() / mask.sum()

                testLosses[morphIdx][-1][0] += autoencoder_loss.item() / numBatchesPerTrainingStep
                testLosses[morphIdx][-1][1] += final_contrastive_loss.item() / numBatchesPerTrainingStep
            testLosses[morphIdx][-1] /= numTestingBatches-1
            s += testLosses[morphIdx][-1].mean()

    lr_scheduler.step(s)

    for morphIdx in trainingIdxs:
        print('Idx {} | Test {} : {}'.format(
            morphIdx, np.round(testLosses[morphIdx][-1][0], decimals=3), np.round(testLosses[morphIdx][-1][1], decimals=3)), flush=True)
    
    stepLoss = None

    for batch in range(0, numTrainingBatches-1, numBatchesPerTrainingStep):
                
        t0 = time.time()
        
        for morphIdx in trainingIdxs:
            numNodes = (X_train[morphIdx].shape[1] - 5) // 2
            trainLosses[morphIdx].append(np.zeros(2))
        
        
        for batchOffset in range(numBatchesPerTrainingStep):
                        
            if batch + batchOffset >= numTrainingBatches - 1:
                break
                
            for morphIdx in trainingIdxs:
                encoder_graph = env[morphIdx].get_graph()._get_dgl_graph()
                decoder_graph = env[morphIdx].get_graph()._get_dgl_graph()
                                
                current_states = X_train[morphIdx][(batch+batchOffset) * batch_size:(batch+batchOffset+1)*batch_size]
                next_states = Y_train[morphIdx][(batch+batchOffset) * batch_size:(batch+batchOffset+1)*batch_size]
                random_indexes = np.random.choice(X_train[trainingIdxs[0]].shape[0], size=current_states.shape[0], replace=False)
                random_states = X_train[morphIdx][random_indexes]
                
                encoderInput = torch.cat((current_states, next_states, random_states), dim=0).to(device)
                latent_states = encoderGNN(encoder_graph, encoderInput)

                current_state_reconstruction = decoderGNN(decoder_graph, latent_states[:, 0:current_states.shape[0], :])
                current_state_reconstruction[:, :, 0:5] = current_state_reconstruction[:, :, 0:5].mean(dim=0)
                autoencoder_loss = criterion(encoder_graph.ndata['input'][:, 0:batch_size, :7], current_state_reconstruction).mean()
                
                # Calculate 2-norm for positive/sequential samples over the data dimension - result is of dimension (nodes, batch_size)
                sequential_distances = torch.norm(latent_states[:, 0:batch_size, :] - latent_states[:, batch_size:batch_size * 2, :], p=None, dim=2)
                # Calculate 2-norm for negative/random samples over data dimension - result is of dimension (nodes, batch_size)
                random_distances = torch.norm(latent_states[:, 0:batch_size, :] - latent_states[:, 2 * batch_size: 3 * batch_size, :], p=None, dim=2)
                # Calculate contrastive loss for each entry - result is of dimension (nodes, batch_size)
                contrastive_loss = torch.max(zeroTensor, sequential_distances - random_distances + minDistanceSeqAndRand)
                # get 0-1 matrix which is True if entry is not 0
                mask = contrastive_loss != 0
                # Compute average over nonzero entries in batch, result will be scalar
                final_contrastive_loss = contrastive_loss.sum() / mask.sum()

                trainLosses[morphIdx][-1][0] += autoencoder_loss.item() / numBatchesPerTrainingStep
                trainLosses[morphIdx][-1][1] += final_contrastive_loss.item() / numBatchesPerTrainingStep

                # stepLoss = autoencoder_loss
                stepLoss = autoencoder_loss + final_contrastive_loss
                stepLoss /= (len(trainingIdxs) * numBatchesPerTrainingStep)

                stepLoss.backward()
                        
        
        if batch % 200 == 0:
            print('Batch {} in {:.1f}s'.format(batch, time.time() - t0), flush=True)
            for morphIdx in trainingIdxs:
                print('Idx {} | Train {:.3f} : {:.3f}'.format(morphIdx, trainLosses[morphIdx][-1][0], trainLosses[morphIdx][-1][1]), flush=True)

        optimizer.step()        
        optimizer.zero_grad()
        
        # Dereference variables to release memory
        stepLoss = None
        encoder_graph = None
        decoder_graph = None
        encoderInput = None
        latent_states = None
        current_state_reconstruction = None
        autoencoder_loss = None
        contrastive_loss_1 = None
        contrastive_loss_2 = None
        torch.cuda.empty_cache()

        t_final = time.time() - t0

    print('Epoch {} finished in {:.1f}'.format(epoch, time.time() - epoch_t0), flush=True)
    save_weights_and_graph(save_dir=save_dir)






