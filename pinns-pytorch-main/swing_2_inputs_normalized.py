from pinns_v2.model import MLP, ModifiedMLP
from pinns_v2.components import ComponentManager, ResidualComponent, ICComponent, SupervisedComponent
from pinns_v2.rff import GaussianEncoding 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from pinns.train import train
from pinns_v2.train import train
from pinns_v2.gradient import _jacobian, _hessian
from pinns_v2.dataset import DomainDatasetRandom, ICDatasetRandom, DomainSupervisedDataset, DomainDatasetSeq, ICDatasetSeq
from torch import sin, pi, sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 3500
num_inputs = 3 #delta0, omega0, t
num_outputs = 2 #delta, omega

csi = 3
B = 1
Vg = 1.2
Ve = 5
P = 3
k = 1

u_min = -0.21
u_max = 0.0
delta_min = 0.0
delta_max = 2*pi
omega_min = 0.0
omega_max = 2*pi
t_f = 2


params = {
    "u_min": u_min,
    "u_max": u_max,
    "delta_min": delta_min,
    "delta_max": delta_max,
    "omega_min": omega_min,
    "omega_max": omega_max,
    "t_f": t_f
}

def hard_constraint(x_in, y_out):
    tau = x_in[-1].reshape(-1,1)

    
    t = tau * t_f
      
    omega = x_in[1] * (omega_max - omega_min) + omega_min
    delta = x_in[0] * (delta_max - delta_min) + delta_min

    u1 = y_out[0]*t + delta
    u2 = y_out[1]*t + omega
    
    return torch.flatten(torch.hstack((u1,u2)))


def pde_fn_del(model, sample):
    
    J, d = _jacobian(model, sample, i = 1 , j = 2)
    
    ddX = J
    
    omega = model(sample)[1]
    delta = model(sample)[0]
    
    return  k*omega + B*Vg*Ve*sin(delta) - P + csi*ddX / t_f


def pde_fn_omega(model, sample):
    
    J, d = _jacobian(model, sample , i = 0 , j = 2)

    dX = J

    omega = model(sample)[1] 

    return dX  / t_f - omega


batchsize = 500
learning_rate = 0.001203836177626117

print("Building Domain Dataset")
domainDataset = DomainDatasetRandom([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 10000, period = 3)
#print("Building IC Dataset")
#icDataset = ICDatasetRandom([0.0]*(num_inputs-1),[10.0]*(num_inputs-1), 10000, period = 3)
#print("Building Domain Supervised Dataset")
#dsdDataset = DomainSupervisedDataset("C:\\Users\\desan\\Documents\\Wolfram Mathematica\\file.csv", 1000)
print("Building Validation Dataset")
validationDataset = DomainDatasetRandom([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], batchsize, shuffle = False)
#print("Building Validation IC Dataset")
#validationicDataset = ICDatasetRandom([0.0]*(num_inputs-1),[10.0]*(num_inputs-1), batchsize, shuffle = False)

model = MLP([num_inputs] + [308]*8 + [num_outputs], nn.SiLU, hard_constraint, p_dropout=0.0, encoding = None)


component_manager = ComponentManager()
r = ResidualComponent([pde_fn_del, pde_fn_omega], domainDataset)
component_manager.add_train_component(r)

#ic = ICComponent([ic_fn_vel], icDataset)
#component_manager.add_train_component(ic)
#d = SupervisedComponent(dsdDataset)
#component_manager.add_train_component(d)
r = ResidualComponent([pde_fn_del, pde_fn_omega], validationDataset)
component_manager.add_validation_component(r)
#ic = ICComponent([ic_fn_vel], validationicDataset)
#component_manager.add_validation_component(ic)


def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

model = model.apply(init_normal)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


data = {
    "name": "swing_norm_3_inputs.0_2000epochs_1",
    #"name": "prova",
    "model": model,
    "epochs": epochs,
    "batchsize": batchsize,
    "optimizer": optimizer,
    "scheduler": None,
    "component_manager": component_manager,
    "additional_data": params
}

train(data, output_to_file=True)
