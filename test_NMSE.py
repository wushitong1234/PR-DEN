import sys
sys.path.append(".") 
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PR_DEN
from utils import load_CEdataset, compute_NMSE, load_checkpoint

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device = ', device)


num_measurements = 512
num_antennas = 1024
dirname = './dataset/'
A = loadmat(dirname + 'matrixA')['A']
block_line1 = np.hstack((np.real(A), -np.imag(A)))
block_line2 = np.hstack((np.imag(A), np.real(A)))
A = torch.from_numpy(np.vstack((block_line1, block_line2))).float()
testing_SNR = np.array([0,5,10,15,20])
testing_channels = 'testing_channel.mat'
channel_path = dirname + testing_channels
testing_NMSE = np.zeros(len(testing_SNR))
lat_layers = 4
contraction_factor = 0.99
eps = 1e-2
max_depth = 25
structure = 'ResNet'
num_channels = 64

net =PR_DEN(A=A, lat_layers=lat_layers, contraction_factor=contraction_factor,
                 eps=eps, max_depth=max_depth, structure=structure, num_channels=num_channels)

for i in range(len(testing_SNR)):
    if testing_SNR[i] < 10:
        checkpoint_PATH = './checkpoints/PR_DEN_ResNet_weights_0to10dB.pth'
    else:
        checkpoint_PATH = './checkpoints/PR_DEN_ResNet_weights_10to20dB.pth'
    net = load_checkpoint(net, checkpoint_PATH, device).to(device)
    net.eval()

    testing_measurements = 'testing_measurements_' + str(testing_SNR[i]) + 'dB.mat'
    measurement_path = dirname + testing_measurements
    measurements, channels, _ = load_CEdataset(measurement_path, channel_path)
    measurements = measurements.to(device)
    channels = channels.to(device)

    measurements = measurements[0:500,:]
    channels = channels[0:500,:]

    channels_pred = net(measurements)
    testing_NMSE[i] = compute_NMSE(channels_pred, channels)

print(testing_NMSE)


