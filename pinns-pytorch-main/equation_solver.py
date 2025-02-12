from scipy.integrate import odeint
import numpy as np
from math import pi, sin
import matplotlib.pyplot as plt

def fun(y, t, csi, B, Vg, Ve, P, k):
    delta,omega = y
    #k*dX + B*Vg*Ve*sin(model(sample)) - P +csi*ddX 
    dydt = [omega, (-k*omega - B*Vg*Ve*sin(delta) + P)/csi]
    return dydt

csi = 3
B = 1
Vg = 1.2
Ve = 5
P = 3
k = 1

y0 = [pi/5, 1]
t = np.linspace(0,2, 1000)
sol = odeint(fun, y0, t, args=(csi, B, Vg, Ve, P, k))

plt.plot(t, sol[:, 0], 'b', label='delta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
