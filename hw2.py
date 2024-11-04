# import numpy as np
# from scipy.integrate import odeint
# from scipy.integrate import RK45
# import matplotlib.pyplot as plt

# def shoot(y, x, Ep):
#     return [y[1], (np.square(x) - Ep) * y[0]]

# tol = 1e-4  # define a tolerance level 
# col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
# xrange = [-4, 4]
# xshoot =  np.arange(-4, 4.1, 0.1)

# eigenfunctions = np.empty([81,5])
# eigenvalues = np.empty([5,])

# Ep_start = 0  # beginning value of energy eigenvalue
# for mode in range(1, 6):  # begin mode loop
#     Ep = Ep_start  # initial value of energy eigenvalue
#     dEp = 0.01  # default step size in energy eigenvalue
#     for _ in range(1000):  # begin convergence loop for beta
#         y0 = [1, 1 * np.sqrt(16-Ep)]
#         phi = odeint(shoot, y0, xshoot, args=(Ep,))
#         yf = -1 * np.sqrt(16-Ep) * phi[-1,0] 
#         if abs(phi[-1, 1] - yf) < tol:  # check for convergence
#             print(Ep)  # write out eigenvalue
#             break  # get out of convergence loop
#         if (-1) ** (mode + 1) * (phi[-1, 1] - yf) > 0:
#             Ep += dEp
#         else:
#             Ep -= dEp / 2
#             dEp /= 2
    
#     eigenvalues[mode-1,] = Ep
#     Ep_start = Ep + 0.1  # after finding eigenvalue, pick new start
#     norm = np.trapz(phi[:, 0] * phi[:, 0], xshoot)  # calculate the normalization
#     eigenfunctions[:, mode-1] = np.abs(phi[:,0] / np.sqrt(norm))
    
#     plt.plot(xshoot, phi[:, 0] / np.sqrt(norm), col[mode - 1])  # plot modes
    
    
# plt.show()  # end mode loop

# A1 = eigenfunctions
# A2 = eigenvalues

# print(np.shape(eigenfunctions))
# print(np.shape(eigenvalues))

# print(eigenfunctions)

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def shoot(y, x, Ep):
    return [y[1], (np.square(x) - Ep) * y[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
xrange = [-4, 4]
xshoot =  np.arange(-4, 4.1, 0.1)

eigenfunctions = np.empty([81,5])
eigenvalues = np.empty([5,])

Ep_start = 0  # beginning value of energy eigenvalue
for mode in range(1, 6):  # begin mode loop
    Ep = Ep_start  # initial value of energy eigenvalue
    dEp = 0.01  # default step size in energy eigenvalue
    for _ in range(1000):  # begin convergence loop for beta
        y0 = [1, 1 * np.sqrt(16-Ep)]
        phi = odeint(shoot, y0, xshoot, args=(Ep,))
        yf = -1 * np.sqrt(16-Ep) * phi[-1,0] 
        if abs(phi[-1, 1] - yf) < tol:  # check for convergence
            print(Ep)  # write out eigenvalue
            break  # get out of convergence loop
        if (-1) ** (mode + 1) * (phi[-1, 1] - yf) > 0:
            Ep += dEp
        else:
            Ep -= dEp / 2
            dEp /= 2
    
    eigenvalues[mode-1,] = Ep
    Ep_start = Ep + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(phi[:, 0] * phi[:, 0], xshoot)  # calculate the normalization
    eigenfunctions[:, mode-1] = np.abs(phi[:,0] / np.sqrt(norm))
    
    plt.plot(xshoot, phi[:, 0] / np.sqrt(norm), col[mode - 1])  # plot modes
    
    
plt.show()  # end mode loop

A1 = eigenfunctions
A2 = eigenvalues

print(np.shape(eigenfunctions))
print(np.shape(eigenvalues))

print(eigenfunctions)
