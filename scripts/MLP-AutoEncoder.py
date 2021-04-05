
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
            self.layers.append(nn.BatchNorm1d(num_features=(hidden_sizes[0])))
        self.layers.append(nn.ReLU())
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if with_batch_norm:
                self.layers.append(nn.BatchNorm1d(num_features=(hidden_sizes[i+1])))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_sizes[len(hidden_sizes) - 1], self.output_size))
        
        if activation is not None:
            self.layers.append(activation())
            
    def forward(self, x):
        out = x
        
        for layer in self.layers:
            out = layer(out)
            
        return out

def save_weights_and_graph(save_dir):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.save(encoder.state_dict(), save_dir + 'encoder.pt')
    torch.save(decoder.state_dict(), save_dir + 'decoder.pt')

    np.save(save_dir + '/trainLosses.npy', np.stack(trainLosses))
    np.save(save_dir +'/testLosses.npy', np.stack(testLosses))


idx = 3
save_dir = 'models/MLP-inverseDynamics/no-contrastive/' + str(idx) + '/'

prefix = 'datasets/' + str(idx) + '/'
states = np.load(prefix + 'states_array.npy')
actions = np.load(prefix + 'actions_array.npy')
rewards = np.load(prefix + 'rewards_array.npy')
next_states = np.load(prefix + 'next_states_array.npy')
dones = np.load(prefix + 'dones_array.npy')



X = states
Y = next_states
permutation = np.random.permutation(X.shape[0])
X = X[permutation]
X_test = torch.from_numpy(X[:100000]).float()
X_train= torch.from_numpy(X[100000:]).float()
Y = Y[permutation]
Y_test = torch.from_numpy(Y[:100000]).float()
Y_train= torch.from_numpy(Y[100000:]).float()


# In[26]:


num_nodes = (X_train.shape[1]-5) // 2
batch_size = 1024
inputSize = states.shape[1]
latentSize = num_nodes * 2
print('State size decreased from {} to {}'.format(X_train.shape[1], num_nodes * 2))
outputSize = inputSize
hidden_sizes = [256, 512, 256]
with_batch_norm = True
activation = nn.Tanh
minDistanceSeqAndRand = 0.25

encoder = Network(input_size=inputSize, output_size=latentSize, hidden_sizes=hidden_sizes, with_batch_norm=with_batch_norm).to(device)
decoder = Network(input_size=latentSize, output_size=inputSize, hidden_sizes=hidden_sizes, with_batch_norm=with_batch_norm).to(device)


# In[27]:


lr = 5e-4
optimizer = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, verbose=True, min_lr=1e-5)
criterion = nn.MSELoss(reduction='none')


# In[28]:


numTrainingBatches = int(np.ceil(X_train.shape[0] / batch_size))
numTestingBatches = int(np.ceil(X_test.shape[0] / batch_size))

print('numTrainingBatches', numTrainingBatches)
print('numTestingBatches', numTestingBatches)

zeroTensor = torch.zeros([1]).to(device)

trainLosses = []
testLosses = []


for epoch in range(50):
    
    print('Starting Epoch {}'.format(epoch))
    epoch_t0 = time.time()
    
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    Y_train = Y_train[permutation]

    with torch.no_grad():
        
        testLosses.append(np.zeros(2))
        for batch in range(0, numTestingBatches-1):

            current_states = X_test[batch * batch_size:(batch+1)*batch_size]
            next_states = Y_test[batch * batch_size:(batch+1)*batch_size]
            random_indexes = np.random.choice(X_test.shape[0],size=batch_size, replace=False)
            random_states = X_test[random_indexes]

            encoderInput = torch.cat((current_states, next_states, random_states), dim=0).to(device)

            latent_states = encoder(encoderInput)
            normalized_latent_states = latent_states / torch.sqrt(1e-8 + (latent_states ** 2).sum(dim=-1)).unsqueeze(-1)
            current_state_reconstruction = decoder(normalized_latent_states[0:batch_size])

            autoencoder_loss = criterion(encoderInput[:batch_size], current_state_reconstruction).mean()
            # Calculate 2-norm for positive/sequential samples over the data dimension - result is of dimension (nodes, batch_size)
            sequential_distances = torch.norm(normalized_latent_states[:batch_size] - normalized_latent_states[batch_size:batch_size * 2], p=None, dim=-1)
            # Calculate 2-norm for negative/random samples over data dimension - result is of dimension (nodes, batch_size)
            random_distances = torch.norm(normalized_latent_states[:batch_size] - normalized_latent_states[2 * batch_size: 3 * batch_size], p=None, dim=-1)
            # Calculate contrastive loss for each entry - result is of dimension (nodes, batch_size)
            contrastive_loss = torch.max(zeroTensor, sequential_distances - random_distances + minDistanceSeqAndRand)
            # get 0-1 matrix which is True if entry is not 0
            mask = contrastive_loss != 0
            # Compute average over nonzero entries in batch, result will be scalar
            final_contrastive_loss = contrastive_loss.sum() / (mask.sum() + 1e-8)

            testLosses[-1][0] += autoencoder_loss.item()
            testLosses[-1][1] += final_contrastive_loss.item()
        testLosses[-1] /= numTestingBatches-1

    print('Test {:.3f} : {:.3f}'.format(testLosses[-1][0], testLosses[-1][1]))
    
    for batch in range(0, numTrainingBatches-1):
                
        t0 = time.time()
        
        trainLosses.append(np.zeros(2))
                        
        current_states = X_train[batch * batch_size:(batch+1)*batch_size]
        next_states = Y_train[batch * batch_size:(batch+1)*batch_size]
        random_indexes = np.random.choice(X_train.shape[0],size=batch_size, replace=False)
        random_states = X_train[random_indexes]

        encoderInput = torch.cat((current_states, next_states, random_states), dim=0).to(device)

        latent_states = encoder(encoderInput)
        normalized_latent_states = latent_states / torch.sqrt(1e-8 + (latent_states ** 2).sum(dim=-1)).unsqueeze(-1)
        current_state_reconstruction = decoder(normalized_latent_states[0:batch_size])

        autoencoder_loss = criterion(encoderInput[:batch_size], current_state_reconstruction).mean()
        # Calculate 2-norm for positive/sequential samples over the data dimension - result is of dimension (nodes, batch_size)
        sequential_distances = torch.norm(normalized_latent_states[:batch_size] - normalized_latent_states[batch_size:batch_size * 2], p=None, dim=-1)
        # Calculate 2-norm for negative/random samples over data dimension - result is of dimension (nodes, batch_size)
        random_distances = torch.norm(normalized_latent_states[:batch_size] - normalized_latent_states[2 * batch_size: 3 * batch_size], p=None, dim=-1)
        # Calculate contrastive loss for each entry - result is of dimension (nodes, batch_size)
        contrastive_loss = torch.max(zeroTensor, sequential_distances - random_distances + minDistanceSeqAndRand)
        # get 0-1 matrix which is True if entry is not 0
        mask = contrastive_loss != 0
        # Compute average over nonzero entries in batch, result will be scalar
        final_contrastive_loss = contrastive_loss.sum() / mask.sum()
        
        # stepLoss = autoencoder_loss
        stepLoss = autoencoder_loss + final_contrastive_loss
        
        trainLosses[-1][0] += autoencoder_loss.item()
        trainLosses[-1][1] += final_contrastive_loss.item()

        
        optimizer.zero_grad()
        stepLoss.backward()
        optimizer.step()
                        
        
        if batch % 200 == 0:
            print('Batch {} in {:.1f}s | Train {:.3f} : {:.3f}'.format(batch, time.time() - t0, trainLosses[-1][0], trainLosses[-1][1]))
        
    print('Epoch {} finished in {:.1f}'.format(epoch, time.time() - epoch_t0))
    save_weights_and_graph(save_dir)


# In[ ]:




