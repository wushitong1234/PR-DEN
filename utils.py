import sys
sys.path.append(".") 
import torch
import torch.nn as nn
import time
import numpy as np
from scipy.io import loadmat
from prettytable import PrettyTable
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_CEdataset(measurement_path, channel_path):
    measurements = loadmat(measurement_path)['y']
    measurements = np.vstack((np.real(measurements), np.imag(measurements))).T
    measurements = torch.from_numpy(measurements).float()

    channels = loadmat(channel_path)['H']
    channels = np.vstack((np.real(channels), np.imag(channels))).T
    channels = torch.from_numpy(channels).float()

    noise_levels = loadmat(measurement_path)['sigma_squared'].T
    noise_levels = torch.from_numpy(noise_levels).float()
    return measurements, channels, noise_levels

class CEdataset():
    def __init__(self, dirname, train=True):
        super(CEdataset, self).__init__()
        if train == True:
            training_measurements = 'training_measurements_10to20dB.mat' # change this when training the low SNR dataset
            training_channels = 'training_channels.mat'   
            self.measurements, self.channels, self.noise_levels = load_CEdataset(f"{dirname}"+training_measurements, f"{dirname}"+training_channels)
        else:
            validation_measurements = 'training_measurements_10to20dB.mat'
            validation_channels = 'validation_channel.mat'   
            self.measurements, self.channels, self.noise_levels = load_CEdataset(f"{dirname}"+validation_measurements,f"{dirname}"+ validation_channels)

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, index):
        measurement = self.measurements[index]
        channel = self.channels[index]
        noise_level = self.noise_levels[index]
        return measurement, channel, noise_level

def dataloaders(dirname,  train_batch_size, test_batch_size=None):
    train_dataset = CEdataset(dirname,  train=True)
    test_dataset = CEdataset(dirname, train=False)

    if test_batch_size is None:
        test_batch_size = train_batch_size

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size, drop_last=True)
    return train_loader, test_loader

def get_stats(net, test_loader, criterion):

    test_loss = 0.0

    with torch.no_grad():
        for y_test, h_test, sigma_squared_test in test_loader:
            h_test = h_test.to(net.device())
            y_test = y_test.to(net.device())
            sigma_squared_test = sigma_squared_test.to(net.device())
            batch_size = y_test.shape[0]

            h_predict = net(y_test)
            batch_loss = criterion(h_predict, h_test)
            test_loss += batch_size * batch_loss

    test_loss /= len(test_loader.dataset)
    test_NMSE = compute_NMSE(h_predict, h_test)
    
    return test_loss, test_NMSE

def model_params(net):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['TOTAL', num_params])
    return table

def compute_NMSE(h_predict, h_label):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    h_predict = h_predict.to(device)
    h_label = h_label.to(device)
    batch_size = h_predict.shape[0]
    NMSE= 0.0
    for i in range(batch_size):
        numerator = torch.square(torch.norm((h_predict[i,:] - h_label[i,:]), p=2))
        denominator = torch.square(torch.norm(h_label[i,:], p=2))
        NMSE += numerator / denominator
    NMSE = NMSE / batch_size
    NMSE = 10 * torch.log10(NMSE)
    return NMSE

class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, h_predict, h_label):
        batch_size = h_predict.shape[0]
        NMSE = 0.0
        for i in range(batch_size):
            numerator = torch.square(torch.norm((h_predict[i, :] - h_label[i, :]), p=2))
            denominator = torch.square(torch.norm(h_label[i, :], p=2))
            NMSE += numerator / denominator
        loss = NMSE / batch_size
        return loss

def train_PR_DEN(net, max_epochs, lr_scheduler, train_loader,
          validation_loader, optimizer, criterion, save_dir='./results'):
    
    fmt = '[{:3d}/{:3d}]: train - ({:6.2f} dB, {:6.2e}), validation - ({:6.2f} dB, '
    fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

    depth_ave = 0.0
    best_validation_NMSE = 0.0

    total_time = 0.0
    time_hist = []
    validation_loss_hist = []
    validation_NMSE_hist = []
    train_loss_hist = []
    train_NMSE_hist = []

    print(net)
    print(model_params(net))

    for epoch in range(max_epochs):
        time.sleep(0.5)  
        loss_ave = 0.0
        train_NMSE = 0.0
        epoch_start_time = time.time()
        tot = len(train_loader)

        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))

            for _, (measurements, channels, _) in enumerate(train_loader):
                channels = channels.to(net.device())
                measurements = measurements.to(net.device())
                batch_size = measurements.shape[0]

                net.train()
                
                optimizer.zero_grad()
                channels_pred = net(measurements)

                depth_ave = 0.99 * depth_ave + 0.01 * net.depth
                output = None

                output = criterion(channels_pred, channels)
                loss_val = output.detach().cpu().numpy() * batch_size
                loss_ave += loss_val
                output.backward()
                optimizer.step()

                train_NMSE = compute_NMSE(channels_pred, channels)
                tepoch.update(1)
                tepoch.set_postfix(train_loss="{:5.2e}".format(loss_val / batch_size),
                                   train_NMSE="{:f}".format(train_NMSE),
                                   depth="{:5.1f}".format(net.depth))

        loss_ave = loss_ave / len(train_loader.dataset)

        net.eval()

        validation_loss, validation_NMSE = get_stats(net,
                                         validation_loader,
                                         criterion)

        validation_loss_hist.append(validation_loss)
        validation_NMSE_hist.append(validation_NMSE)
        train_loss_hist.append(loss_ave)
        train_NMSE_hist.append(train_NMSE)

        epoch_end_time = time.time()
        time_epoch = epoch_end_time - epoch_start_time

        time_hist.append(time_epoch)
        total_time += time_epoch

        print(fmt.format(epoch+1, max_epochs, train_NMSE, loss_ave,
                         validation_NMSE, validation_loss, depth_ave,
                         optimizer.param_groups[0]['lr'],
                         time_epoch))
        
        net.train()

        if validation_NMSE < best_validation_NMSE:
            best_validation_NMSE = validation_NMSE
            state = {
                'test_loss_hist': validation_loss_hist,
                'test_NMSE_hist': validation_NMSE_hist,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler
            }
            file_name = save_dir + net.name() + '_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)


        lr_scheduler.step()
        epoch_start_time = time.time()
    return net

def load_checkpoint(model, checkpoint_PATH, device):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH, map_location=torch.device(device))
        model.load_state_dict(model_CKPT['net_state_dict'])
        print('Checkpoint has been loaded!')
    return model
