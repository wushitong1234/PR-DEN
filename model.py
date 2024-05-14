import torch
import torch.nn as nn
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

measurements = torch.tensor
step_size = torch.tensor
latent_solution = torch.tensor
noise_level = torch.tensor
final_solution = torch.tensor

class PR_DEN(nn.Module):
    def __init__(self,
                 A,
                 lat_layers=3,
                 contraction_factor=0.99,
                 eps=1.0e-2,
                 max_depth=15,
                 sigma_HPR=1,
                 structure='ResNet',
                 num_channels=64):
        super(PR_DEN, self).__init__()
        self.A = A.to(device)   
        self.sigma_HPR=sigma_HPR
        self.W_pinv = torch.from_numpy(np.linalg.pinv(A)).to(device)   
        self.M_inv=torch.inverse(self.sigma_HPR*torch.eye(self.A.shape[1]).to(device)+torch.mm(self.W_pinv, self.A)).to(device)   
        self.step = self.A.shape[1] / (torch.trace(torch.mm(self.W_pinv, self.A))).to(device)    
        self.W_pinv_mul_step = (self.step * self.W_pinv).to(device)   
        self._lat_layers = lat_layers   
        self.gamma = contraction_factor    
        self.eps = eps   
        self.max_depth = max_depth    
        self.structure = structure    
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.unflatten = nn.Unflatten(1, (8,16,16))  
        self.depth = 0.0
        self.use_layer_norm = True  
        self.use_LS_initialization = True    
        self.input_convs = nn.Conv2d(8, num_channels, kernel_size=3, stride=1, padding=(1,1), bias=True)
        if self.use_layer_norm == True:
            self.input_layer_norm = nn.LayerNorm([num_channels,16,16])
        self.latent_convs = nn.ModuleList([nn.Sequential(
                                           nn.Conv2d(in_channels=num_channels,
                                                     out_channels=num_channels,
                                                     kernel_size=3, stride=1,
                                                     padding=(1, 1)),
                                           self.relu,
                                           nn.Conv2d(in_channels=num_channels,
                                                     out_channels=num_channels,
                                                     kernel_size=3, stride=1,
                                                     padding=(1, 1)),
                                           self.relu)
                                           for _ in range(lat_layers)])   
        if self.use_layer_norm == True:
            self.latent_layer_norm = nn.ModuleList([nn.LayerNorm([num_channels,16,16])
                                                for _ in range(lat_layers)]) 
        self.output_convs = nn.Sequential(nn.Conv2d(in_channels=num_channels,
                                                    out_channels=num_channels,
                                                    kernel_size=1, stride=1),
                                          self.leaky_relu,
                                          nn.Conv2d(in_channels=num_channels,
                                                    out_channels=8,
                                                    kernel_size=1, stride=1)) 

    def name(self):
        if self.structure == 'ResNet':
            return 'PR_DEN_ResNet'
        else:
            print("\nWarning: unsupported backbone network...\n")

    def device(self):
        device = next(self.parameters()).data.device
        return device

    def initialize_solution(self, y: measurements):
        batch_size = y.shape[0]
        if self.use_LS_initialization == True:
            h_init = torch.matmul(self.W_pinv,y.unsqueeze(-1)).squeeze(-1)
        else:
            h_init = torch.zeros(batch_size, self.A.shape[1], device=self.device())
        return h_init


    def Prox(self, y: measurements, h: latent_solution) -> latent_solution: 

        batch_size = y.shape[0]

        if self.structure == 'ResNet':
            h = self.unflatten(h)

            h = self.leaky_relu(self.input_convs(h))
            if self.use_layer_norm == True:
                h = self.input_layer_norm(h)

            for idx, conv in enumerate(self.latent_convs):
                res = conv(h)   
                h = h + res
                if self.use_layer_norm == True:
                    h = self.latent_layer_norm[idx](h)

            h = self.output_convs(h)

            h = h.view(batch_size,-1)

            h = self.gamma * h

        else:
            print("\nWarning: unsupported backbone network...\n")

        return h    

    def PR_forward(self, y, eta_0, eta_HPR2, k):
        # s = torch.matmul(self.M_inv, (self.sigma_HPR * torch.matmul(self.A.t(), y.t()).to(device)+eta_HPR2.t().to(device))).t()
        s = torch.matmul(self.M_inv, (self.sigma_HPR * torch.matmul(self.W_pinv, y.t()).to(device)+eta_HPR2.t().to(device))).t() #
        
        w = eta_HPR2 + self.sigma_HPR* s.to(device)
        
        p = self.Prox(y, (1/self.sigma_HPR*(2*self.sigma_HPR*s-eta_HPR2)))

        x = eta_HPR2 + self.sigma_HPR*p.to(device) -2* self.sigma_HPR  * s 

        v = eta_HPR2 + 2*self.sigma_HPR*(p.to(device) - s.to(device))
        
        eta_HPR2 = 1/(k+2)*eta_0+ (k+1)/(k+2)*v.to(device)

        return p, eta_HPR2 

    def forward(self, y: measurements, depth_warning=False):
        with torch.no_grad():
            self.depth = 0.0
            h = self.initialize_solution(y)
            x_dual=h.to(device)
            y_dual=x_dual.to(device)
            eta_0=2*y_dual-x_dual
            h_prev = np.Inf * torch.ones(h.shape, device=self.device())
            termination = False
            eta_HPR2=eta_0
            while not termination and self.depth < self.max_depth:
                h_prev = h.clone()
                h, eta_HPR2 = self.PR_forward(y,eta_0,eta_HPR2, self.depth)    # original
                res_norm = torch.max(torch.norm(h - h_prev, dim=1))
                self.depth += 1.0
                termination = (res_norm <= self.eps)

        attach_gradients = self.training
        if attach_gradients:
            h = self.PR_forward(y, eta_0, eta_HPR2, self.depth)[0]
            return h
        else:
            return h.detach()
