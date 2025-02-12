from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
from pinns.model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns.train import train
from pinns.dataset import DomainDataset, ICDataset
import json
from torch import sin, pi
from scipy.integrate import odeint



def fun(y, t, csi, B, Vg, Ve, P, k):
    delta,omega = y
    dydt = [omega, (-k*omega - B*Vg*Ve*np.sin(delta) + P)/csi]
    return dydt

def equation_solver(y0,t_f):
    csi = 3
    B = 1
    Vg = 1.2
    Ve = 5
    P = 3
    k = 1

    t = np.linspace(0,t_f, 1000)
    sol = odeint(fun, y0, t, args=(csi, B, Vg, Ve, P, k))
    return sol


def hard_constraint(x_in, y_out):
    tau = x_in[:,-1]
    
    t = tau * t_f
    
    omega = x_in[:,1] * (omega_max - omega_min) + omega_min
    delta = x_in[:,0] * ( delta_max - delta_min) + delta_min
    
    U1 = y_out[:,0]*t + delta
    U2 = y_out[:,1]*t + omega

    return [U1, U2]


def compose_input(t,delta_0,omega_0):
    
    X = np.hstack((np.ones_like(t)*delta_0,np.ones_like(t)*omega_0))
    X = np.hstack((X,t))
    
    X = torch.Tensor(X).to(torch.device("cuda:0")).requires_grad_()

    return X


if __name__ == "__main__":
    name = "output"
    experiment_name = "good_model"
    current_file = os.path.abspath(__file__)
    output_dir = os.path.join(os.path.dirname(current_file), name)
    output_dir = os.path.join(output_dir, experiment_name)

    model_dir = os.path.join(output_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'model.pt')

    output_dir = "./"

    model = torch.load(model_path)
    x = torch.randn(1, 2).to(torch.device("cuda:0"))


    model.train(False)
    
    delta_min = 0.0
    delta_max = 2*pi
    omega_min = 0
    omega_max = 2*pi
    t_f = 2


    time = np.linspace(0, t_f, num=1000, endpoint=True).reshape(-1, 1)

    delta_0 = pi/6
    omega_0 = 2

    X = compose_input(time / t_f, delta_0/delta_max , omega_0/omega_max)
    
    print(X)

    pred = model(X)
    

    y0 = [delta_0 , omega_0]
    real = equation_solver(y0,t_f)
    
    delta_real = real[:, 0]
    omega_real = real[:,1]

    delta = pred[0].cpu().detach().numpy()

    plt.plot(time,delta_real, "r" ,  label = "real_delta")
    plt.plot(time,delta, "b" ,label = "delta")

    omega = pred[1].cpu().detach().numpy()

    plt.plot(time,omega_real, "g" ,label="real_omega")
    plt.plot(time,omega,"black", label="omega")
    plt.legend()
    
    plt.figure()
    
    plt.title("Error")
    plt.plot(time, abs(delta_real - delta) ,label="Delta error" )
    plt.plot(time, abs(omega_real - omega), label="Omega error")
    plt.legend()
    plt.show()