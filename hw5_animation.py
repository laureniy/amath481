import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
from scipy.fftpack import fft2, ifft2
import os
from matplotlib.colors import Normalize
import imageio.v2 as imageio  # Ensure you have imageio installed (`pip install imageio`)

# Setting spatial domain and time
N = 256
size = N*N
nu = 0.001
dt = 0.5
tf = 10
timespan = (0,tf)
times = np.arange(0.0, tf, dt)
L = 10
span = np.linspace(-L, L, N+1)
x = span[0:N]
y = span[0:N]
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# making A, B, C matrices
e0 = np.zeros((size, 1))  # vector of zeros
e1 = np.ones((size, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, N+1):
    e2[N*j-1] = 0  # overwrite every m^th value with zero
    e4[N*j-1] = 1  # overwirte every m^th value with one

e3 = np.zeros_like(e2)
e3[1:size] = e2[0:size-1]
e3[0] = e2[size-1]

e5 = np.zeros_like(e4)
e5[1:size] = e4[0:size-1]
e5[0] = e4[size-1]

Adiagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
Aoffsets = [-(size-N), -N, -N+1, -1, 0, 1, N-1, N, (size-N)]

matA = spdiags(Adiagonals, Aoffsets, size, size).toarray()

A1 = matA/(dx**2)

Bdiagonals = [e1.flatten(), -1*e1.flatten(), e1.flatten(), -1*e1.flatten()]
Boffsets = [-(size-N), -N, N, (size-N)]

matB = spdiags(Bdiagonals, Boffsets, size, size).toarray()

A2 = matB/(2*dx)

e6 = np.copy(e1) # copy the one vector
e7 = np.copy(e1) # copy the one vector
e8 = np.copy(e0) # copy the zero vector
e9 = np.copy(e0) # copy the zero vector

for j in range(1, N):
    e6[N*j-1] = 0  # overwrite every m^th value with zero
    e7[N*j] = 0  # overwrite every m^th value with zero
    e8[N*j-1] = 1
    e9[N*j] = 1

e7[N*(N-1)] = 0
e8[-1] = 1
e9[0] = 1

Cdiagonals = [e9.flatten(), -1*e6.flatten(), e7.flatten(), -1*e8.flatten()]
Coffsets = [-(N-1), -1, 1, N-1]

matC = spdiags(Cdiagonals, Coffsets, size, size).toarray()

A3 = matC/(2*dy)

A = A1
B = A2
C = A3

# Setting initial condition
omega0 = np.exp((-(X**2))-((Y**2)/20))
omega0opp = 5*(np.exp((-((X+3)**2)/8)-((Y**2)/5))) - 5*(np.exp((-((X-3)**2)/8)-((Y**2)/5)))
omega0double = (5*np.exp((-(X-3)**2)-(Y**2)) + (5*np.exp((-(X+3)**2)-(Y**2))))
omega0collide = (np.exp((-(X**2)/3)-(((Y-7)**2)/3))) - (np.exp((-(X**2)/3)-(((Y+7)**2)/3)))
omega0everywhere = 0.5*(np.exp((-(X**2)/3)-(((Y-7)**2)/3))) - 0.5*(np.exp((-(2*(X+5)**2)/6)-(((Y+6)**2)/6))) + 0.8*(np.exp((-((X-3)**2)/8)-(((Y+3)**2)/8))) - 1.3*(np.exp((-(2*(X-7)**2)/5)-((Y**2)/7))) + (np.exp((-(2*(X+6)**2)/5)-(((Y-3)**2)/10)))

# Setting wave numbers for fft
kx = (np.pi / L) * np.concatenate((np.arange(0, N/2), np.arange(-N/2, 0)))
kx[0] = 1e-6
ky = (np.pi / L) * np.concatenate((np.arange(0, N/2), np.arange(-N/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

omega_evolution = np.zeros((size, int(tf/dt)))
def rhsa(t, omegaflat, A, B, C, nu):
    omega = omegaflat.reshape((N, N))
    omegat = fft2(omega)
    psit = -omegat/K
    psi = ifft2(psit)
    psiflat = np.real(psi.reshape(size))
    rhs = np.real(nu*A.dot(omegaflat) - B.dot(psiflat)*C.dot(omegaflat) + C.dot(psiflat)*B.dot(omegaflat))
    return rhs

sola = solve_ivp(rhsa, timespan, omega0opp.reshape(size), t_eval = times, args=(A, B, C, nu))
omegafa = sola.y
plt.figure(figsize = (36, 24))
for ja, tia in enumerate(times):
    omega_evolution[:size, ja] = omegafa[:size,ja]
#     omegaia = np.real(omegafa[:size,ja]).reshape((N,N))
#     plt.subplot(8, 5, ja + 1)
#     plt.pcolor(x, y, omegaia, shading='interp')
#     plt.title(f'Time: {tia}')
#     plt.colorbar()

# plt.tight_layout()
# plt.show()

import os
from matplotlib.colors import Normalize
import imageio.v2 as imageio  # Ensure you have imageio installed (`pip install imageio`)

# Create output directory for frames
output_dir = "vorticity_frames"
os.makedirs(output_dir, exist_ok=True)

# Create a colormap for consistency
colormap = 'plasma'

# Loop over time steps and save frames
for j, t in enumerate(times):
    omegai = np.real(omega_evolution[:, j].reshape((N, N)))
    
    # Plot vorticity in position space
    plt.figure(figsize=(6, 6))
    plt.pcolor(x, y, omegai, shading='auto', cmap=colormap, norm=Normalize(vmin=-1, vmax=1))
    plt.title(f"Time: {t:.2f}")
    plt.colorbar(label="Vorticity")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Save the frame
    frame_path = os.path.join(output_dir, f"frame_{j:04d}.png")
    plt.savefig(frame_path)
    plt.close()

# Combine frames into an animated GIF
gif_path = "vorticity_evolution_omegaopp.gif"
frames = [imageio.imread(os.path.join(output_dir, f)) for f in sorted(os.listdir(output_dir)) if f.endswith(".png")]
imageio.mimsave(gif_path, frames, fps=5)  # Adjust fps as needed

print(f"Animation saved to {gif_path}")
plt.show()