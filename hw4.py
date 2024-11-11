import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.sparse import spdiags

N = 8
size = N*N
L = 10
span = np.linspace(-L, L, N+1)
xspan = span[0:N]
yspan = span[0:N]
dx = xspan[1] - xspan[0]
dy = yspan[1] - yspan[0]

e0 = np.zeros((size, 1))  # vector of zeros
e1 = np.ones((size, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, N+1):
    e2[N*j-1] = 0  # overwrite every m^th value with zero
    e4[N*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:size] = e2[0:size-1]
e3[0] = e2[size-1]

e5 = np.zeros_like(e4)
e5[1:size] = e4[0:size-1]
e5[0] = e4[size-1]

# Place diagonal elements
Adiagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
Aoffsets = [-(size-N), -N, -N+1, -1, 0, 1, N-1, N, (size-N)]

matA = spdiags(Adiagonals, Aoffsets, size, size).toarray()

# Plot matrix structure
plt.figure(5)
plt.spy(matA)
plt.title('A Matrix Structure')
plt.show()

A1 = (matA + np.zeros((64,64)))/(dx**2)
print(A1)

Bdiagonals = [e1.flatten(), -1*e1.flatten(), e1.flatten(), -1*e1.flatten()]
Boffsets = [-(size-N), -N, N, (size-N)]

matB = spdiags(Bdiagonals, Boffsets, size, size).toarray()

plt.figure(5)
plt.spy(matB)
plt.title('B Matrix Structure')
plt.show()

A2 = (matB + np.zeros((64,64)))/(2*dx)
print(A2)

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

plt.figure(5)
plt.spy(matC)
plt.title('C Matrix Structure')
plt.show()

A3 = (matC + np.zeros((64,64)))/(2*dy)
print(A3)








