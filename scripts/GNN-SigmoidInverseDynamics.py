import numpy as np
import torch.autograd
import time
import torch.optim as optim
import torch.nn as nn
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
        activation=None,
        with_sigmoid=False
    ):
        super(Network, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList()
        self.with_sigmoid = with_sigmoid

        self.layers.append(nn.Linear(self.input_size, hidden_sizes[0]))
        if with_batch_norm:
            self.layers.append(nn.LayerNorm(normalized_shape=(hidden_sizes[0])))
        self.layers.append(nn.ReLU())
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if with_batch_norm:
                self.layers.append(nn.LayerNorm(normalized_shape=(hidden_sizes[i+1])))
            self.layers.append(nn.ReLU())
        
        
        self.outputLayer = nn.Linear(hidden_sizes[-1], self.output_size)
        self.outputActivation = activation
        
        
        if with_sigmoid:
            self.sigmoidLayer = nn.Linear(hidden_sizes[-1], 1)
            
    def forward(self, x):
        out = x
        
        for layer in self.layers:
            out = layer(out)
        
        final_output = self.outputLayer(out)
        
        if self.outputActivation is not None:
            final_output = self.outputActivation(final_output)
        
        if self.with_sigmoid:
            
            sigmoid_output = self.sigmoidLayer(out).sigmoid()
            final_output = torch.cat((final_output, sigmoid_output), dim=-1)
            
        return final_output

class GraphNeuralNetwork(nn.Module):
    def __init__(
        self,
        inputNetwork,
        messageNetwork,
        updateNetwork,
        outputNetwork,
        numMessagePassingIterations,
        withInputNetwork = True
    ):
        
        super(GraphNeuralNetwork, self).__init__()
                
        self.inputNetwork = inputNetwork
        self.messageNetwork = messageNetwork
        self.updateNetwork = updateNetwork
        self.outputNetwork = outputNetwork
        
        self.numMessagePassingIterations = numMessagePassingIterations
        self.withInputNetwork = withInputNetwork
        
    def inputFunction(self, nodes):
        return {'state' : self.inputNetwork(nodes.data['input'])}
    
    def messageFunction(self, edges):
        
        batchSize = edges.src['state'].shape[1]
        edgeData = edges.data['feature'].repeat(batchSize, 1).T.unsqueeze(-1)
        nodeInput = edges.src['input']
        
        return {'m' : self.messageNetwork(torch.cat((edges.src['state'], edgeData, nodeInput), -1))}
    
    def updateFunction(self, nodes):
        return {'state': self.updateNetwork(torch.cat((nodes.data['m_hat'], nodes.data['state']), -1))}
    
    def outputFunction(self, nodes):
        
        return {'output': self.outputNetwork(nodes.data['state'])}


    def forward(self, graph, state):
        
        self.update_states_in_graph(graph, state)
        
        if self.withInputNetwork:
            graph.apply_nodes(self.inputFunction)
        
        for messagePassingIteration in range(self.numMessagePassingIterations):
            graph.update_all(self.messageFunction, dgl.function.mean('m', 'm_hat'), self.updateFunction)
        
        graph.apply_nodes(self.outputFunction)
        
        output = graph.ndata['output']
        output = torch.transpose(output, dim0=0, dim1=1)
        
        actions = output[:, :, 0].squeeze(-1)
        sigmoids = output[:, :, 1].squeeze(-1).mean(-1)
                
        return actions, sigmoids
    
    def update_states_in_graph(self, graph, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        numGraphFeature = 6
        numGlobalStateInformation = 5
        numLocalStateInformation = 2
        numStateVar = state.shape[1] // 2
        globalInformation = torch.cat((state[:, 0:5], state[:, numStateVar:numStateVar+5]), -1)
        
        numNodes = (numStateVar - 5) // 2

        nodeData = torch.empty((numNodes, state.shape[0], numGraphFeature + 2 * numGlobalStateInformation + 2 * numLocalStateInformation)).to(device)
        for nodeIdx in range(numNodes):

            # Assign global features from graph
            nodeData[nodeIdx, :, :6] = graph.ndata['feature'][nodeIdx]
            # Assign local state information
            nodeData[nodeIdx, :, 16] = state[:, 5 + nodeIdx]
            nodeData[nodeIdx, :, 17] = state[:, 5 + numNodes + nodeIdx]
            nodeData[nodeIdx, :, 18] = state[:, numStateVar + 5 + nodeIdx]
            nodeData[nodeIdx, :, 19] = state[:, numStateVar + 5 + numNodes + nodeIdx]

        # Assdign global state information
        nodeData[:, :, 6:16] = globalInformation
        
        if self.withInputNetwork:
            graph.ndata['input'] = nodeData        
        
        else:
            graph.ndata['state'] = nodeData

def save_weights_and_graph(save_dir):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.save(gnn.state_dict(), save_dir + 'gnn.pt')

    for morphIdx in trainingIdxs:
        np.save(save_dir + '/actionTrainLosses.npy', np.stack(actionTrainLosses[morphIdx]))
        np.save(save_dir +'/sigmoidTrainLosses.npy', np.stack(sigmoidTrainLosses[morphIdx]))
        np.save(save_dir +'/actionTestLosses.npy', np.stack(actionTestLosses[morphIdx]))
        np.save(save_dir +'/sigmoidTestLosses.npy', np.stack(sigmoidTestLosses[morphIdx]))


# idx = 0
# trainingIdxs = [idx]

trainingIdxs = [0,1,2,3,4,5]

save_dir = 'models/inverseDynamics-multi-with-sigmoid/'
# save_dir = 'models/inverseDynamics-with-sigmoid/' + str(idx) + '/'

states = {}
actions = {}
rewards = {}
next_states = {}
dones = {}
env = {}

for morphIdx in trainingIdxs:

    prefix = 'datasets/{}/'.format(morphIdx)
    
    states[morphIdx] = np.load(prefix + 'states_array.npy')
    actions[morphIdx] = np.load(prefix + 'actions_array.npy')
    rewards[morphIdx] = np.load(prefix + 'rewards_array.npy')
    next_states[morphIdx] = np.load(prefix + 'next_states_array.npy')
    dones[morphIdx] = np.load(prefix + 'dones_array.npy')
    
    env[morphIdx] = HalfCheetahGraphEnv(None)
    env[morphIdx].set_morphology(morphIdx)


X_test = {}
X_train = {}
Y_test = {}
Y_train = {}

for morphIdx in trainingIdxs:
    X = np.concatenate((states[morphIdx], next_states[morphIdx]), -1)
    Y = actions[morphIdx]
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    X_test[morphIdx] = torch.from_numpy(X[:100000]).float()
    X_train[morphIdx] = torch.from_numpy(X[100000:]).float()
    Y = Y[permutation]
    Y_test[morphIdx] = torch.from_numpy(Y[:100000]).float()
    Y_train[morphIdx] = torch.from_numpy(Y[100000:]).float()

hidden_sizes = [256, 256]

inputSize = 20
stateSize = 64
messageSize = 64
outputSize = 1
numMessagePassingIterations = 6
batch_size = 2048
with_batch_norm = True
numBatchesPerTrainingStep = 1

inputNetwork = Network(inputSize, stateSize, hidden_sizes, with_batch_norm)
messageNetwork = Network(stateSize + inputSize + 1, messageSize, hidden_sizes, with_batch_norm, nn.Tanh())
updateNetwork = Network(stateSize + messageSize, stateSize, hidden_sizes, with_batch_norm)
outputNetwork = Network(stateSize, outputSize, hidden_sizes, with_batch_norm, nn.Tanh(), with_sigmoid=True)

gnn = GraphNeuralNetwork(inputNetwork, messageNetwork, updateNetwork, outputNetwork, numMessagePassingIterations).to(device)

lr = 5e-4
optimizer = optim.Adam(itertools.chain(inputNetwork.parameters(), messageNetwork.parameters(), updateNetwork.parameters(), outputNetwork.parameters())
                       , lr=lr, weight_decay=0)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, verbose=True, min_lr=5e-6, threshold=1e-2)
l2Loss = nn.MSELoss(reduction='none')
binaryLoss = nn.BCELoss()
zeroTensor = torch.zeros([batch_size]).to(device)
oneTensor = torch.ones([batch_size]).to(device)
binaryLossWeighing = 1e-4

numTrainingBatches = int(np.ceil(X_train[trainingIdxs[0]].shape[0] / batch_size))
numTestingBatches = int(np.ceil(X_test[trainingIdxs[0]].shape[0] / batch_size))

actionTrainLosses = {}
actionTestLosses = {}

sigmoidTrainLosses = {}
sigmoidTestLosses = {}
for morphIdx in trainingIdxs:
    actionTrainLosses[morphIdx] = []
    actionTestLosses[morphIdx] = []
    
    sigmoidTrainLosses[morphIdx] = []
    sigmoidTestLosses[morphIdx] = []

for epoch in range(15):
    
    print('Starting Epoch {}'.format(epoch))
    epoch_t0 = time.time()
    
    for morphIdx in trainingIdxs:
        permutation = np.random.permutation(X_train[morphIdx].shape[0])
        X_train[morphIdx] = X_train[morphIdx][permutation]
        Y_train[morphIdx] = Y_train[morphIdx][permutation]

    with torch.no_grad():

        for morphIdx in trainingIdxs:
            numNodes = ((X_train[morphIdx].shape[1] // 2) - 5) // 2
            actionTestLosses[morphIdx].append(torch.zeros(numNodes))
            sigmoidTestLosses[morphIdx].append(torch.zeros(4))
            for batch_ in range(0, numTestingBatches-1):

                g1 = env[morphIdx].get_graph()._get_dgl_graph()
                g2 = env[morphIdx].get_graph()._get_dgl_graph()
                                
                x = X_test[morphIdx][batch_ * batch_size:(batch_+1)*batch_size].to(device)
                y = Y_test[morphIdx][batch_ * batch_size:(batch_+1)*batch_size].to(device)
                
                predicted_actions, predicted_sigmoids = gnn(g1, x)
                
                actionsLoss = l2Loss(predicted_actions, y)
                sigmoidLoss = binaryLossWeighing * binaryLoss(predicted_sigmoids, oneTensor)
                
                actionTestLosses[morphIdx][-1] += actionsLoss.mean(dim=0).cpu().detach()
                sigmoidTestLosses[morphIdx][-1][0] += sigmoidLoss.item()
                sigmoidTestLosses[morphIdx][-1][1] += torch.eq(oneTensor, torch.round(predicted_sigmoids)).sum().item() / float(batch_size)

                totalForwardLoss = actionsLoss.mean() + sigmoidLoss

                xHalfLength = int(x.shape[-1] / 2)
                predicted_actions, predicted_sigmoids = gnn(g2, torch.cat((x[:, xHalfLength:], x[:, :xHalfLength]), -1))
                
                backwardLoss = binaryLossWeighing * binaryLoss(predicted_sigmoids, zeroTensor)

                sigmoidTestLosses[morphIdx][-1][2] += backwardLoss.item()
                sigmoidTestLosses[morphIdx][-1][3] += torch.eq(zeroTensor, torch.round(predicted_sigmoids)).sum().item() / float(batch_size)

            actionTestLosses[morphIdx][-1] /= numTrainingBatches-1
            sigmoidTestLosses[morphIdx][-1] /= numTrainingBatches-1

    s = 0
    for morphIdx in trainingIdxs:
        print('Test Idx {} | Actions Loss {} \nSigmoid L&A {}\n'.format(
            morphIdx, np.round(actionTestLosses[morphIdx][-1], decimals=3), np.round(sigmoidTestLosses[morphIdx][-1], decimals=3)))
        s += actionTestLosses[morphIdx][-1].mean()

    lr_scheduler.step(s)

    for batch in range(0, numTrainingBatches-1, numBatchesPerTrainingStep):
                
        t0 = time.time()
        
        for morphIdx in trainingIdxs:
            numNodes = ((X_train[morphIdx].shape[1] // 2) - 5) // 2
            actionTrainLosses[morphIdx].append(torch.zeros(numNodes))
            sigmoidTrainLosses[morphIdx].append(torch.zeros(4))
        
        optimizer.zero_grad()       
        
        for batchOffset in range(numBatchesPerTrainingStep):

            if batch + batchOffset >= numTrainingBatches - 1:
                break
                
            for morphIdx in trainingIdxs:
                
                g1 = env[morphIdx].get_graph()._get_dgl_graph()
                g2 = env[morphIdx].get_graph()._get_dgl_graph()
                                
                x = X_train[morphIdx][(batch+batchOffset) * batch_size:(batch+batchOffset+1)*batch_size].to(device)
                y = Y_train[morphIdx][(batch+batchOffset) * batch_size:(batch+batchOffset+1)*batch_size].to(device)
                
                predicted_actions, predicted_sigmoids = gnn(g1, x)
                
                actionsLoss = l2Loss(predicted_actions, y)
                sigmoidLoss = binaryLossWeighing * binaryLoss(predicted_sigmoids, oneTensor)
                
                actionTrainLosses[morphIdx][-1] += actionsLoss.mean(dim=0).cpu().detach()
                sigmoidTrainLosses[morphIdx][-1][0] += sigmoidLoss.item()
                sigmoidTrainLosses[morphIdx][-1][1] += torch.eq(oneTensor, torch.round(predicted_sigmoids)).sum().item() / float(batch_size)

                totalForwardLoss = actionsLoss.mean() + sigmoidLoss
                totalForwardLoss.backward()

                xHalfLength = int(x.shape[-1] / 2)
                predicted_actions, predicted_sigmoids = gnn(g2, torch.cat((x[:, xHalfLength:], x[:, :xHalfLength]), -1))
                
                backwardLoss = binaryLossWeighing * binaryLoss(predicted_sigmoids, zeroTensor)
                backwardLoss.backward()

                sigmoidTrainLosses[morphIdx][-1][2] += backwardLoss.item()
                sigmoidTrainLosses[morphIdx][-1][3] += torch.eq(zeroTensor, torch.round(predicted_sigmoids)).sum().item() / float(batch_size)
        
        for morphIdx in trainingIdxs:
            actionTestLosses[morphIdx][-1] /= numBatchesPerTrainingStep
            sigmoidTestLosses[morphIdx][-1] /= numBatchesPerTrainingStep

        optimizer.step()        
        
        if batch % 200 == 0:
            print('Batch {} in {}s'.format(batch, np.round(time.time() - t0, decimals=1)))
            for morphIdx in trainingIdxs:
                print('Train Idx {} | Actions Loss {} \nSigmoid L&A {}\n'.format(
                    morphIdx, np.round(actionTrainLosses[morphIdx][-1], decimals=3), np.round(sigmoidTrainLosses[morphIdx][-1], decimals=3)))

        t_final = time.time() - t0

    print('Epoch {} finished in {}'.format(epoch, np.round(time.time() - epoch_t0, decimals=1)))
    save_weights_and_graph(save_dir)
