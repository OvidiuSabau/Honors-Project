import matplotlib.pyplot as plt
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
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if with_batch_norm:
                self.layers.append(nn.LayerNorm(normalized_shape=(hidden_sizes[i + 1])))
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
        output = torch.transpose(output, dim0=0, dim1=1).squeeze(-1)

        return output
    
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

        trainingLosses = np.stack(actionTrainLosses[morphIdx])
        testingLosses = np.stack(actionTestLosses[morphIdx])

        np.save(save_dir + str(morphIdx) + '-actionTrainLosses.npy', trainingLosses)
        np.save(save_dir + str(morphIdx) + '-actionTestLosses.npy', testingLosses)

        try:
            plt.close(fig)
        except:
            pass

        fig = plt.figure()

        num_epochs = testingLosses.shape[0]
        num_batches_per_epoch = trainingLosses.shape[0] // num_epochs
        trainingLosses = trainingLosses.reshape(num_epochs, num_batches_per_epoch, trainingLosses.shape[1])
        trainingLosses = trainingLosses.mean(1)

        for node in range(trainingLosses.shape[1]):
            plt.plot(np.arange(trainingLosses.shape[0]), np.log10(trainingLosses[:, node]), c='blue')
            plt.plot(np.arange(trainingLosses.shape[0]), np.log10(testingLosses[:, node]), c='red')
            plt.legend(['Training', 'Testing'])
        fig.savefig(save_dir + str(morphIdx) + 'losses-graph.png')

idx = 5
trainingIdxs = [idx]

# trainingIdxs = [0,1,2,3,4,5]

save_dir = 'models/new/inverseDynamics-single/' + str(idx) + '-attempt-5/'
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
    # X = torch.from_numpy(np.concatenate((states[morphIdx], next_states[morphIdx]), axis=-1)).to(torch.float32)
    X = torch.from_numpy(states[morphIdx]).repeat(1, 2).to(torch.float32)
    X[:, X.shape[1] // 2:] -= next_states[morphIdx]
    Y = torch.from_numpy(actions[morphIdx]).to(torch.float32)
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    X_test[morphIdx] = X[:100000]
    X_train[morphIdx] = X[100000:]
    Y = Y[permutation]
    Y_test[morphIdx] = Y[:100000]
    Y_train[morphIdx] = Y[100000:]

hidden_sizes = [256, 256]

inputSize = 20
stateSize = 64
messageSize = 64
outputSize = 1
numMessagePassingIterations = 6
batch_size = 2048
with_batch_norm = True
numBatchesPerTrainingStep = 8

inputNetwork = Network(inputSize, stateSize, hidden_sizes, with_batch_norm)
messageNetwork = Network(stateSize + inputSize + 1, messageSize, hidden_sizes, with_batch_norm, nn.Tanh)
updateNetwork = Network(stateSize + messageSize, stateSize, hidden_sizes, with_batch_norm)
outputNetwork = Network(stateSize, outputSize, hidden_sizes, with_batch_norm, nn.Tanh)

gnn = GraphNeuralNetwork(inputNetwork, messageNetwork, updateNetwork, outputNetwork, numMessagePassingIterations).to(device)

print(gnn.load_state_dict(torch.load('models/new/inverseDynamics-single/5-attempt-2/gnn.pt')))
lr = 1e-5
optimizer = optim.Adam(itertools.chain(inputNetwork.parameters(), messageNetwork.parameters(), updateNetwork.parameters(), outputNetwork.parameters()), lr=lr, weight_decay=0)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, verbose=True, min_lr=5e-6, threshold=1e-3)
l2Loss = nn.MSELoss(reduction='none')

numTrainingBatches = int(np.ceil(X_train[trainingIdxs[0]].shape[0] / batch_size))
numTestingBatches = int(np.ceil(X_test[trainingIdxs[0]].shape[0] / batch_size))

actionTrainLosses = {}
actionTestLosses = {}

for morphIdx in trainingIdxs:
    actionTrainLosses[morphIdx] = []
    actionTestLosses[morphIdx] = []

for epoch in range(200):
    
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
            for batch_ in range(numTestingBatches):

                g1 = env[morphIdx].get_graph()._get_dgl_graph()
                g2 = env[morphIdx].get_graph()._get_dgl_graph()
                                
                x = X_test[morphIdx][batch_ * batch_size:(batch_+1)*batch_size].to(device)
                y = Y_test[morphIdx][batch_ * batch_size:(batch_+1)*batch_size].to(device)
                
                predicted_actions = gnn(g1, x)
                
                actionsLoss = l2Loss(predicted_actions, y)

                actionTestLosses[morphIdx][-1] += actionsLoss.mean(dim=0).cpu().detach()

            actionTestLosses[morphIdx][-1] /= numTestingBatches
    s = 0
    for morphIdx in trainingIdxs:
        print('Test Idx {} | Actions Loss {} \n'.format(
            morphIdx, np.round(actionTestLosses[morphIdx][-1], decimals=3)))
        s += actionTestLosses[morphIdx][-1].mean()

    lr_scheduler.step(s)

    for batch in range(0, numTrainingBatches, numBatchesPerTrainingStep):
                
        t0 = time.time()
        
        for morphIdx in trainingIdxs:
            numNodes = ((X_train[morphIdx].shape[1] // 2) - 5) // 2
            actionTrainLosses[morphIdx].append(torch.zeros(numNodes))

        optimizer.zero_grad()
        
        for batchOffset in range(numBatchesPerTrainingStep):

            if batch + batchOffset >= numTrainingBatches - 1:
                break
                
            for morphIdx in trainingIdxs:
                
                g1 = env[morphIdx].get_graph()._get_dgl_graph()
                g2 = env[morphIdx].get_graph()._get_dgl_graph()
                                
                x = X_train[morphIdx][(batch+batchOffset) * batch_size:(batch+batchOffset+1)*batch_size].to(device)
                y = Y_train[morphIdx][(batch+batchOffset) * batch_size:(batch+batchOffset+1)*batch_size].to(device)
                
                predicted_actions = gnn(g1, x)
                
                actionsLoss = l2Loss(predicted_actions, y).mean(dim=0)
                actionTrainLosses[morphIdx][-1] += actionsLoss.cpu().detach() / numBatchesPerTrainingStep

                actionsLoss = actionsLoss.mean() / numBatchesPerTrainingStep
                actionsLoss.backward()

        optimizer.step()        
        
        if batch % 100 == 0:
            print('Batch {} in {}s'.format(batch, np.round(time.time() - t0, decimals=1)))
            for morphIdx in trainingIdxs:
                print('Train Idx {} | Actions Loss {} \n'.format(
                    morphIdx, np.round(actionTrainLosses[morphIdx][-1], decimals=3)))

        t_final = time.time() - t0
    print('Epoch {} finished in {}'.format(epoch, np.round(time.time() - epoch_t0, decimals=1)))
    save_weights_and_graph(save_dir)
