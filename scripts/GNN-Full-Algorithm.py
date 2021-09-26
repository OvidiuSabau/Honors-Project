import numpy as np
import torch.autograd
import time
import torch.nn as nn
import torch.nn.functional as F
import dgl
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
from graphenvs import HalfCheetahGraphEnv


class EncoderDecoderSubNetwork(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            with_batch_norm=False,
            activation=None
    ):
        super(EncoderDecoderSubNetwork, self).__init__()
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

class EncoderDecoderGNN(nn.Module):
    def __init__(
            self,
            inputNetwork,
            messageNetwork,
            updateNetwork,
            outputNetwork,
            numMessagePassingIterations,
            encoder=True
    ):

        super(EncoderDecoderGNN, self).__init__()

        self.inputNetwork = inputNetwork
        self.messageNetwork = messageNetwork
        self.updateNetwork = updateNetwork
        self.outputNetwork = outputNetwork

        self.numMessagePassingIterations = numMessagePassingIterations
        self.encoder = encoder

    def inputFunction(self, nodes):
        return {'state': self.inputNetwork(nodes.data['input'])}

    def messageFunction(self, edges):

        batchSize = edges.src['state'].shape[1]
        edgeData = edges.data['feature'].repeat(batchSize, 1).T.unsqueeze(-1)
        nodeInput = edges.src['input']

        #         print(edges.src['state'].shape)
        #         print(nodeInput.shape)
        return {'m': self.messageNetwork(torch.cat((edges.src['state'], edgeData, nodeInput), -1))}

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

            nodeData = torch.empty(
                (numNodes, batch_size, numGraphFeature + numGlobalStateInformation + numLocalStateInformation)).to(
                device)

            nodeData[:, :, 0:numGlobalStateInformation] = globalInformation
            for nodeIdx in range(numNodes):
                # Assign local state information
                nodeData[nodeIdx, :, numGlobalStateInformation] = state[:, 5 + nodeIdx]
                nodeData[nodeIdx, :, numGlobalStateInformation + 1] = state[:, 5 + numNodes + nodeIdx]
                # Assign global features from graph
                nodeData[nodeIdx, :, numGlobalStateInformation + 2: numGlobalStateInformation + 2 + numGraphFeature] = \
                    graph.ndata['feature'][nodeIdx]

            graph.ndata['input'] = nodeData

        else:
            numNodes, batchSize, inputSize = state.shape
            nodeData = torch.empty((numNodes, batchSize, inputSize + 6)).to(device)
            nodeData[:, :, :inputSize] = state
            nodeData[:, :, inputSize: inputSize + 6] = graph.ndata['feature'].unsqueeze(dim=1).repeat_interleave(
                batchSize, dim=1)
            #             for nodeIdx in range(numNodes):
            #                 nodeData[nodeIdx, :, inputSize : inputSize + 6] = graph.ndata['feature'][nodeIdx]

            graph.ndata['input'] = nodeData

class InverseDynamicsSubNetwork(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            with_batch_norm=False,
            activation=None,
            with_sigmoid=False
    ):
        super(InverseDynamicsSubNetwork, self).__init__()
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
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if with_batch_norm:
                self.layers.append(nn.LayerNorm(normalized_shape=(hidden_sizes[i + 1])))
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

class InverseDynamicsGNN(nn.Module):
    def __init__(
            self,
            inputNetwork,
            messageNetwork,
            updateNetwork,
            outputNetwork,
            numMessagePassingIterations,
            withInputNetwork=True
    ):

        super(InverseDynamicsGNN, self).__init__()

        self.inputNetwork = inputNetwork
        self.messageNetwork = messageNetwork
        self.updateNetwork = updateNetwork
        self.outputNetwork = outputNetwork

        self.numMessagePassingIterations = numMessagePassingIterations
        self.withInputNetwork = withInputNetwork

    def inputFunction(self, nodes):
        return {'state': self.inputNetwork(nodes.data['input'])}

    def messageFunction(self, edges):

        batchSize = edges.src['state'].shape[1]
        edgeData = edges.data['feature'].repeat(batchSize, 1).T.unsqueeze(-1)
        nodeInput = edges.src['input']

        return {'m': self.messageNetwork(torch.cat((edges.src['state'], edgeData, nodeInput), -1))}

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
        globalInformation = torch.cat((state[:, 0:5], state[:, numStateVar:numStateVar + 5]), -1)

        numNodes = (numStateVar - 5) // 2

        nodeData = torch.empty((numNodes, state.shape[0],
                                numGraphFeature + 2 * numGlobalStateInformation + 2 * numLocalStateInformation)).to(
            device)
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


def reconstructStateFromDecoderOutput(state):
    numNodes, batchSize, latentSize = state.shape
    finalState = torch.empty(batchSize, 5 + numNodes * 2)
    finalState[:, 0:5] = state[:, :, 0:5].mean(dim=0)
    finalState[:, 5: 5 + numNodes] = state[:, :, 5].T
    finalState[:, 5 + numNodes: 5 + 2 * numNodes] = state[:, :, 6].T

    return finalState


def calculate_action(initial_state, inverseDynamicsNetwork, encoder, decoder, env, alpha_seq, gamma, H, R,
                     batch_size=512, latentSize=2):
    # Initialize list of trajectories with one trajectory containing the initial state
    trajectories = [[initial_state]]

    # Keep track of latest state encodings to avoid having to recompute them multiple times
    # Initialize it to have the encoding of the initial state
    g = env.get_graph()._get_dgl_graph().to(device)
    latest_state_encodings = [encoder(g, initial_state.to(device)).cpu()]

    # Initialize list of trajectory rewards to 0
    cumulative_rewards = [0]

    # Compute length of noise vectors so that distance to resulting latent encodings is within range
    sqrt_alpha_seq = np.sqrt(latentSize * alpha_seq) + 7e-2

    s = 0
    count = 0

    # Iteratively deepen to H time-steps in the future
    for h in range(H):

        # Initialize lists which will hold the trajectories resulting from this iteration
        new_trajectories = []
        new_latest_state_encodings = []
        new_cumulative_rewards = []

        # Expand each trajectory
        for traj_idx, trajectory in enumerate(trajectories):
            # Get the saved encoding for trajectory
            encoding = latest_state_encodings[traj_idx]
            # Create array containing possible neighbour encodings with shape (batch_size, latent_size)
            batch_encodings = encoding.repeat_interleave(batch_size, dim=1)

            # Sample noise uniformly across [-1, 1], with shape (batch_size, latent_size)
            noise = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(batch_size, batch_encodings.shape[-1])))
            # Make each noise vector have length sqrt_alpha_seq
            noise = F.normalize(noise, dim=-1) * sqrt_alpha_seq

            # Apply noise, normalize resulting encodings so they have length 1, then compute resulting states
            batch_encodings += noise
            next_encodings = F.normalize(batch_encodings, dim=-1).to(torch.float32).to(device)
            g = env.get_graph()._get_dgl_graph().to(device)
            next_states = reconstructStateFromDecoderOutput(decoder(g, next_encodings)).to(device)

            # Get array of repeated current states, necessary to prune backward in time states from next_states
            repeated_current_state = trajectory[-1].repeat_interleave(batch_size, dim=0).to(device)
            repeated_current_encoding = encoding.repeat_interleave(batch_size, dim=1).to(device)

            #             print(((repeated_current_encoding - next_encodings) ** 2).mean())
            #             print(((repeated_current_encoding - next_encodings) ** 2).std())

            g = env.get_graph()._get_dgl_graph().to(device)

            actions, sigmoid = inverseDynamicsNetwork(g, torch.cat((repeated_current_state, next_states), dim=-1))

            not_forward_mask = (sigmoid < 0.5).squeeze(-1)

            s += not_forward_mask.sum().item()
            count += 1

            rewards = next_states[:, 0] + not_forward_mask * (-100)
            best_next_states_indices = np.argpartition(rewards.cpu().numpy(), -R)[-R:]

            best_states = list(next_states[best_next_states_indices].cpu().unbind())
            next_trajectories = [trajectory + [x.unsqueeze(0)] for x in best_states]

            new_state_encodings = list(
                next_encodings[:, best_next_states_indices, :].cpu().transpose(0, 1).unsqueeze(-2).unbind())
            next_rewards = list(
                (next_states[best_next_states_indices, 0] * (gamma ** h) + cumulative_rewards[traj_idx]).cpu().unbind())

            new_trajectories.extend(next_trajectories)
            new_latest_state_encodings.extend(new_state_encodings)
            new_cumulative_rewards.extend(next_rewards)

        trajectories = new_trajectories
        cumulative_rewards = new_cumulative_rewards
        latest_state_encodings = new_latest_state_encodings

    best_trajectory_idx = np.argmax(cumulative_rewards)
    actual_next_state = trajectories[best_trajectory_idx][1]

    g = env.get_graph()._get_dgl_graph().to(device)
    actions, sigmoid = inverseDynamicsNetwork(g, torch.cat((initial_state, actual_next_state), dim=-1).to(device))

    #     print(s / count)
    return actions

prefix = 'models/GNN-AutoEncoder/multi-GNN-4-latent-contrastive/'

hidden_sizes = [256, 256]
inputSize = 13
stateSize = 64
messageSize = 64
latentSize = 4
numMessagePassingIterations = 6
with_batch_norm = True


# # Encoder Networks
encoderInputNetwork = EncoderDecoderSubNetwork(inputSize, stateSize, hidden_sizes, with_batch_norm=with_batch_norm)
encoderMessageNetwork = EncoderDecoderSubNetwork(stateSize + inputSize + 1, messageSize, hidden_sizes, with_batch_norm=with_batch_norm, activation=nn.Tanh)
encoderUpdateNetwork = EncoderDecoderSubNetwork(stateSize + messageSize, stateSize, hidden_sizes, with_batch_norm=with_batch_norm)
encoderOutputNetwork = EncoderDecoderSubNetwork(stateSize, latentSize, hidden_sizes, with_batch_norm=with_batch_norm)
encoderGNN = EncoderDecoderGNN(encoderInputNetwork, encoderMessageNetwork, encoderUpdateNetwork, encoderOutputNetwork, numMessagePassingIterations, encoder=True).to(device)

# # Decoder Networks
decoderInputNetwork = EncoderDecoderSubNetwork(latentSize + 6, stateSize, hidden_sizes, with_batch_norm=with_batch_norm)
decoderMessageNetwork = EncoderDecoderSubNetwork(stateSize + latentSize + 7, messageSize, hidden_sizes, with_batch_norm=with_batch_norm, activation=nn.Tanh)
decoderUpdateNetwork = EncoderDecoderSubNetwork(stateSize + messageSize, stateSize, hidden_sizes, with_batch_norm=with_batch_norm)
decoderOutputNetwork = EncoderDecoderSubNetwork(stateSize, 7, hidden_sizes, with_batch_norm=with_batch_norm)
decoderGNN = EncoderDecoderGNN(decoderInputNetwork, decoderMessageNetwork, decoderUpdateNetwork, decoderOutputNetwork, numMessagePassingIterations, encoder=False).to(device)

encoderGNN.load_state_dict(torch.load(prefix + 'encoderGNN.pt'))
decoderGNN.load_state_dict(torch.load(prefix + 'decoderGNN.pt'))

prefix = 'models/inverseDynamics-multi-with-sigmoid/gnn.pt'
inputSize = 20
stateSize = 64
messageSize = 64
outputSize = 1
numMessagePassingIterations = 6
with_batch_norm = True

inputNetwork = InverseDynamicsSubNetwork(inputSize, stateSize, hidden_sizes, with_batch_norm)
messageNetwork = InverseDynamicsSubNetwork(stateSize + inputSize + 1, messageSize, hidden_sizes, with_batch_norm, nn.Tanh())
updateNetwork = InverseDynamicsSubNetwork(stateSize + messageSize, stateSize, hidden_sizes, with_batch_norm)
outputNetwork = InverseDynamicsSubNetwork(stateSize, outputSize, hidden_sizes, with_batch_norm, nn.Tanh(), with_sigmoid=True)

inverseDynamics = InverseDynamicsGNN(inputNetwork, messageNetwork, updateNetwork, outputNetwork, numMessagePassingIterations).to(device)
inverseDynamics.load_state_dict(torch.load(prefix))

idx = 0

env = HalfCheetahGraphEnv(None)
env.set_morphology(idx)
alpha_seq = 0.03
gamma = 0.9
H = 3
R = 3
batch_size = 8192
num_trials = 5
episode_steps = 500

with torch.no_grad():

    print('Starting ({}, {}) and batch_size {}...'.format(H, R, batch_size), end='', flush=True)
    trial_t0 = time.time()
    trial_rewards = []
    for _ in range(num_trials):

        current_state = torch.from_numpy(env.reset()).to(torch.float32).unsqueeze(0)
        t0 = time.time()
        total_reward = 0

        for i in range(episode_steps):
            action = calculate_action(initial_state=current_state, env=env,
                                      inverseDynamicsNetwork=inverseDynamics,
                                      batch_size=batch_size, encoder=encoderGNN, decoder=decoderGNN,
                                      alpha_seq=alpha_seq, gamma=gamma, H=H, R=R)

            new_state, reward, done, _ = env.step(action.cpu().squeeze(0).numpy())
            total_reward += reward
            current_state = torch.from_numpy(new_state).to(torch.float32).unsqueeze(0)

        trial_rewards.append(total_reward)
    print(' Ended in {:.2f}s | Trial rewards mean {:.4f} - std {:.4f}'.format(time.time() - trial_t0, np.mean(trial_rewards), np.std(trial_rewards)), flush=True)
