import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp

#####
#a (fft)
n = 64
N = n*n
beta = 1
d1 = 0.1
d2 = 0.1
m = 1
L = 20

tspan = [0,4]
ts = np.arange(0, 4.5, 0.5)
x2 = np.linspace(-L/2, L/2, n + 1)
x = x2[:n]
y2 = np.linspace(-L/2, L/2, n + 1)
y = y2[:n]
X, Y = np.meshgrid(x, y)

kx = (2 * np.pi / L) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / L) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

#setting initial conditions
u0=np.tanh(np.sqrt(X**2+Y**2))*np.cos(m*np.angle(X+1j*Y)-(np.sqrt(X**2+Y**2)))

u0c = 1 * u0 + 1j * np.zeros((n, n))
u0t = fft2(u0c)
u0tflat = u0t.reshape(N) 
u0tstack = np.hstack([np.real(u0tflat).reshape(N),np.imag(u0tflat).reshape(N)])

v0=np.tanh(np.sqrt(X**2+Y**2))*np.sin(m*np.angle(X+1j*Y)-(np.sqrt(X**2+Y**2)))

v0c = 1 * v0 + 1j * np.zeros((n,n))
v0t = fft2(v0c)
v0tflat = v0t.reshape(N) 
v0tstack = np.hstack([np.real(v0tflat).reshape(N),np.imag(v0tflat).reshape(N)])

#stacking initial conditions
uv0tstack = np.hstack([u0tstack,v0tstack]) # in fourier domain, real and complex separated

def rhsspec(t, uvtstack, beta, d1, d2):
    utflat = uvtstack[0:N] + 1j*uvtstack[N:2*N]
    ut = utflat.reshape((n,n))
    u = ifft2(ut)
    
    vtflat = uvtstack[2*N:3*N] + 1j*uvtstack[3*N:]
    vt = vtflat.reshape((n,n))
    v = ifft2(vt)
    
    A = u**2 + v**2
    
    lmbda = 1-A
    omga = -beta*A
    
    delut = K*ut
    delvt = K*vt
    
    rhsu = (fft2(lmbda*u-omga*v)-d1*delut).reshape(N)
    rhsv = (fft2(omga*u+lmbda*v)-d2*delvt).reshape(N)
    
    return np.hstack([np.real(rhsu), np.imag(rhsu),
                      np.real(rhsv), np.imag(rhsv)])

uvtsolspec = solve_ivp(rhsspec, [0, 4], uv0tstack, t_eval = ts, args=(beta, d1, d2))
uvftstack = uvtsolspec.y.T

ufspec = np.zeros((9, 4096))
uft = np.zeros((9, 4096), dtype = np.complex128)
vfspec = np.zeros((9, 4096))
vft = np.zeros((9, 4096), dtype = np.complex128)
for i, uvft in enumerate (uvftstack):
    u_re = uvft[:N]
    u_im = uvft[N:2*N]

    v_re = uvft[2*N:3*N]
    v_im = uvft[3*N:]
    
    uft[i, :] = u_re + 1j * u_im
    ufspec[i, :] = np.real((ifft2(uft[i,:].reshape((n,n)))).reshape(N))
    vft[i, :] = v_re + 1j * v_im
    vfspec[i, :] = np.real((ifft2(vft[i,:].reshape((n,n)))).reshape(N))
    
ufspec = ufspec.T
uft = uft.T
vfspec = vfspec.T
vft = vft.T

A1 = np.vstack([uft, vft])

#####
#b (Chebychev)
def cheb(N):
    if N==0: 
        D = 0.; x = 1.
    else:
        n = arange(0,N+1)
        x = cos(pi*n/N).reshape(N+1,1) 
        c = (hstack(( [2.], ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
        X = tile(x,(1,N+1))
        dX = X - X.T
        D = dot(c,1./c.T)/(dX+eye(N+1))
        D -= diag(sum(D.T,axis=0))
    return D, x.reshape(N+1)

ncheb = 30
Ncheb = ncheb*ncheb
D, xcheb = cheb(ncheb)
D[0, :] = 0
D[ncheb, :] = 0
D2 = np.dot(D,D)
ID = np.eye(len(D2))
L = (np.kron(ID, D2) + np.kron(D2, ID)) / 100
xcheb = xcheb*10
ycheb = xcheb
Xcheb, Ycheb = np.meshgrid(xcheb, ycheb)

#setting initial conditions
u0cheb = np.tanh(np.sqrt(Xcheb**2+Ycheb**2))*np.cos(m*np.angle(Xcheb+1j*Ycheb)-(np.sqrt(Xcheb**2+Ycheb**2)))
u0chebflat = u0cheb.reshape((ncheb+1)**2)

v0cheb = np.tanh(np.sqrt(Xcheb**2+Ycheb**2))*np.sin(m*np.angle(Xcheb+1j*Ycheb)-(np.sqrt(Xcheb**2+Ycheb**2)))
v0chebflat = v0cheb.reshape((ncheb+1)**2)

#stacking initial conditions
uv0chebflat = np.hstack([u0chebflat, v0chebflat])

def rhscheb(t, uvchebflat, beta, d1, d2, L):
    uflat = uvchebflat[0:(ncheb+1)**2]
    vflat = uvchebflat[(ncheb+1)**2:]
    
    Aflat = uflat**2 + vflat**2
    
    lmbda = 1-Aflat
    omga = -beta*Aflat
    
    rhsu = (lmbda*uflat-omga*vflat)+d1*L.dot(uflat)
    rhsv = (omga*uflat+lmbda*vflat)+d2*L.dot(vflat)
    return np.hstack([rhsu, rhsv])

uvsolcheb = solve_ivp(rhscheb, (0,4), uv0chebflat, t_eval = ts, args=(beta, d1, d2, L))
uvfstack = uvsolcheb.y

ufcheb = np.real(uvfstack[0:(ncheb+1)**2, :])
vfcheb = np.real(uvfstack[(ncheb+1)**2:, :])

A2 = uvfstack
