import torch
import torch.optim as optim
from scipy.io import loadmat
import numpy as np
from model import PR_DEN
from utils import dataloaders, train_PR_DEN
from utils import NMSELoss

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device = ', device)

seed = np.random.randint(1,100)
print(seed)
torch.manual_seed(seed)
dataset_dir = './dataset/'
save_dir = './checkpoints/'
checkpt_path = './checkpoints/'

num_measurements = 512
num_antennas = 1024
dirname = './dataset/'
A = loadmat(dirname + 'matrixA')['A']
block_line1 = np.hstack((np.real(A), -np.imag(A)))
block_line2 = np.hstack((np.imag(A), np.real(A)))
A = torch.from_numpy(np.vstack((block_line1, block_line2))).float()
lat_layers = 4
contraction_factor = 0.99
eps = 1e-2
max_depth = 25
structure = 'ResNet'
num_channels = 64   

net = PR_DEN(A=A, lat_layers=lat_layers, contraction_factor=contraction_factor,
                 eps=eps, max_depth=max_depth, structure=structure, num_channels=num_channels).to(device)

# training setup
max_epochs = 150
learning_rate = 1e-3
weight_decay = 0
optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
loss = NMSELoss()

print('weight_decay = ', weight_decay, ', learning_rate = ', learning_rate,
      ', eps = ', eps, ', max_depth = ', max_depth, 'contraction_factor = ',
      contraction_factor, 'optimizer = Adam')


train_batch_size = 128
validation_batch_size = 2000

train_loader, validation_loader = dataloaders(dataset_dir,  train_batch_size, validation_batch_size)

net = train_PR_DEN(net, max_epochs, lr_scheduler, train_loader,
                    validation_loader, optimizer, loss, save_dir)