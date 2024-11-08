import numpy as np
from scipy.linalg import eig
from scipy.integrate import odeint
from scipy.integrate import RK45
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
#######
# a (rerunning hw2)
print('running part a')
def shoota(xa, ya, Epa):
    return [ya[1], (np.square(xa) - Epa) * ya[0]]

tola = 1e-4  # define a tolerance level 
cola = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
xrangea = [-4, 4.1]
xshoota =  np.arange(-4, 4.1, 0.1)

eigenfunctionsa = np.empty([81,5])
eigenvaluesa = np.empty([5,])

Ep_starta = 0  # beginning value of energy eigenvalue
for modea in range(1, 6):  # begin mode loop
    Epa = Ep_starta  # initial value of energy eigenvalue
    dEpa = 0.01  # default step size in energy eigenvalue
    for _ in range(1000):  # begin convergence loop for beta
        y0a = [1, 1 * np.sqrt(16-Epa)]
        phia = solve_ivp(shoota, xrangea, y0a, t_eval = xshoota, args = (Epa,))
        yfa = -1 * np.sqrt(16-Epa) * phia.y[0, -1] 
        if abs(phia.y[1, -1] - yfa) < tola:  # check for convergence
            print(Epa)  # write out eigenvalue
            break  # get out of convergence loop
        if (-1) ** (modea + 1) * (phia.y[1, -1] - yfa) > 0:
            Epa += dEpa
        else:
            Epa -= dEpa / 2
            dEpa /= 2
    
    eigenvaluesa[modea-1,] = Epa
    Ep_starta = Epa + 0.1  # after finding eigenvalue, pick new start
    norma = np.trapz(phia.y[0] * phia.y[0], xshoota)  # calculate the normalization
    eigenfunctionsa[:, modea-1] = np.abs(phia.y[0] / np.sqrt(norma))
    
    plt.plot(xshoota, phia.y[0] / np.sqrt(norma), cola[modea - 1])  # plot modes
    
    
plt.show()  # end mode loop

A1 = eigenfunctionsa
A2 = eigenvaluesa
print(np.shape(A1))
print(np.shape(A2))
print(A1)

#######
# b
print('running part b')

tolb = 1e-4
xrangeb = [-4, 4]
dxb = 0.1
xb =  np.linspace(xrangeb[0], xrangeb[1], 81)
kb = 1

Ab = np.zeros([79, 79])
for i in range(79):
    Ab[i,i] = 2 + kb*np.square(dxb)*np.square(xb[i+1])
    for i in range(79-1):
        Ab[i,i+1] = -1
        Ab[i+1,i] = -1

Ab[0,0] -= 4/3 # (2/3) + kb*np.square(dxb)*np.square(xb[1])
Ab[0,1] += 1/3 # -2/3
Ab[-1,-2] += 1/3 # -2/3
Ab[-1,-1] -= 4/3 # kb*np.square(dxb)*np.square(xb[78])

Db,Vb = eig(Ab) # Compute eigenvalues/eigenvectors

sorted_indicesb = np.argsort(np.abs(Db))[::-1]
Dbsort = Db[sorted_indicesb]
Vbsort = Vb[:, sorted_indicesb]
Db5 = Dbsort[79-5:79]
Vb5 = Vbsort[:,79-5:79]

Vb5_normalized = np.zeros((81,5))
for i in range(5):
    Vb5_normalized[0, i] = (4/3)*Vb5[0, i] - (1/3)*Vb5[1, i]
    Vb5_normalized[80, i] = (4/3)*Vb5[78, i] - (1/3)*Vb5[77, i]
    Vb5_normalized[1:80, i] = np.real(Vb5[:, i])
    normb = np.trapz(Vb5_normalized[:, i]**2, np.linspace(xrangeb[0], xrangeb[1], 81))
    Vb5_normalized[:, i] = Vb5_normalized[:,i]/np.sqrt(normb)


A3 = np.abs(np.real(Vb5_normalized))
A3 = A3[:, ::-1]

A4 = np.real(Db5 / dxb**2)  # first five eigenvalues
A4 = np.flip(A4)
print(np.shape(A3))
print(np.shape(A4))

plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(np.linspace(xrangeb[0], xrangeb[1], 81), Vb5_normalized[:, i], label=f'Mode {i+1}')

plt.legend()
plt.grid(True)
plt.show()

#######
# c
print('running part c')
tolc = 1e-4  # define a tolerance level 
colc = ['r', 'b']  # eigenfunc colors
linec = ['dashed', 'dotted']
xrangec = [-2, 2]
xshootc =  np.linspace(-2, 2, 41)
i = 0
eigenfunctions_list = [0,0]
eigenvalues_list = [0,0]

for gammac in [0.05, -0.05]:
    def shootc(xc, yc, k, Ep, gamma):
        return [yc[1], (gamma*(np.abs(yc[0])**2) + (k*(xc**2) - Ep)) * yc[0]]
    eigenfunctionsc = np.empty([41,2])
    eigenvaluesc = np.empty([2,])
    kc = 1
    Ep_startc = 0  # beginning value of energy eigenvalue
    A_startc = 1e-6 # beginning shooting "angle"
    for modec in range(1, 3):  # begin mode loop
        Epc = Ep_startc  # initial value of energy eigenvalue
        Ac = A_startc #initial value of shooting "angle"
        dAc = 0.01
        for _ in range(100):
            dEpc = 0.01  # default step size in energy eigenvalue
            for _ in range(1000):  # begin convergence loop for beta and A
                y0c = [Ac, Ac * np.sqrt(4-Epc)]
                sol = solve_ivp(shootc, xrangec, y0c, t_eval = xshootc, args=(kc, Epc, gammac))
                phic = sol.y.T
                yfc = -1 * np.sqrt(4-Epc) * phic[-1,0] # Boundary value
                if abs(phic[1, -1] - yfc) < tolc:  # check for convergence of boundary condition and normalization
                    break  # get out of convergence loop
                if abs(phic[1, -1] - yfc) > tolc:
                    if (-1) ** (modec + 1) * (phic[-1, 1] - yfc) > 0:
                        Epc += dEpc
                    else:
                        Epc -= dEpc / 2
                        dEpc /= 2
        
            normc = np.trapz(phic[:,0] * phic[:,0], xshootc)  # calculate the normalization

            if abs(normc - 1) < tolc:
                eigenvaluesc[modec-1,] = Epc
                print('Ep = ' + str(Epc))
                break
            else:
                if normc < 1:
                    Ac += dAc
                if normc > 1:
                    Ac -= dAc/2
                    dAc/=2

        Ep_startc = Epc + 0.1  # after finding eigenvalue, pick new start
        eigenfunctionsc[:, modec-1] = np.abs(phic[:,0]) / np.sqrt(normc)
        
        plt.plot(xshootc, np.abs(phic[:,0]) / np.sqrt(normc), colc[i])  # plot modes

        eigenfunctions_list[i] = eigenfunctionsc
        eigenvalues_list[i] = eigenvaluesc
    i += 1
plt.show()

A5 = eigenfunctions_list[0]
A6 = eigenvalues_list[0]
A7 = eigenfunctions_list[1]
A8 = eigenvalues_list[1]  
print(np.shape(A5))
print(np.shape(A6))
print(np.shape(A7))
print(np.shape(A8))
print(A7)

#     for _ in range(1000):  # begin convergence loop for beta and A
#         y0c = [Ac, Ac * np.sqrt(16*(kc**2)-Epc)]
#         phic = odeint(shootc, y0c, xshootc, args=(gammac1, Epc, kc))
#         yfc = -1 * np.sqrt(16*(kc**2)-Epc) * phic[-1,0] # Boundary value
#         normc = np.trapz(phic[:, 0] * phic[:, 0], xshootc) # normalization integral
#         if abs(phic[-1, 1] - yfc) < tolc and abs(normc - 1) < tolc:  # check for convergence of boundary condition and normalization
#             print('Ep = '+ str(Epc))  # write out eigenvalue
#             print('A = ' + str(Ac))
#             break  # get out of convergence loop
#         else:
#             if abs(normc - 1) > tolc:
#                 Ac = Ac/np.sqrt(normc)
#             if abs(phic[-1, 1] - yfc) > tolc:
#                 if (-1) ** (modec + 1) * (phic[-1, 1] - yfc) > 0:
#                     Epc += dEpc
#                 else:
#                     Epc -= dEpc / 2
#                     dEpc /= 2
#         # else:
#         #     Ac -= dAc/2
        #       dAc /= 2
        
#         # y0c = [Ac, Ac * np.sqrt(16-Epc)]
#         # phic = odeint(shootc, y0c, xshootc, args=(gammac1, Epc, kc))
#         # yfc = -1 * np.sqrt(16-Epc) * phic[-1,0] # Boundary value
#         # normc = np.trapz(phic[:, 0] * phic[:, 0], xshootc) # normalization integral
#         # if abs(phic[-1, 1] - yfc) < tolc and abs(normc - 1) < tolc:  # check for convergence of boundary condition and normalization
#         #     print('Ep = '+ str(Epc))  # write out eigenvalue
#         #     print('A = ' + str(Ac))
#         #     break  # get out of convergence loop

#         # if (-1) ** (modec + 1) * (phic[-1, 1] - yfc) > 0:
#         #     Epc += dEpc
#         # else:
#         #     Epc -= dEpc / 2
#         #     dEpc /= 2
#     print('converged!')
#     eigenvaluesc1[modec-1,] = Epc
#     Ep_startc = Epc + 0.1  # after finding eigenvalue, pick new start
#     normc = np.trapz(phic[:, 0] * phic[:, 0], xshootc)  # calculate the normalization
#     print('Norm = ' + str(normc))
#     eigenfunctionsc1[:, modec-1] = np.abs(phic[:,0] / np.sqrt(normc))
    
#     plt.plot(xshootc, phic[:, 0] / np.sqrt(normc), colc[modec - 1])  # plot modes
    
    
# plt.show()  # end mode loop

# A5 = eigenfunctionsc1
# A6 = eigenvaluesc1

# print(eigenfunctionsc1)
# print(eigenvaluesc1)

# eigenfunctionsc2 = np.empty([41,2])
# eigenvaluesc2 = np.empty([2,])

# gammac2 = -0.05
# for modec in range(1, 3):  # begin mode loop
#     Epc = Ep_startc  # initial value of energy eigenvalue
#     dEpc = 0.01  # default step size in energy eigenvalue
#     Ac = A_startc #initial value of shooting "angle"
#     for _ in range(1000):  # begin convergence loop for beta and A
#         y0c = [Ac, Ac * np.sqrt(16*(kc**2)-Epc)]
#         phic = odeint(shootc, y0c, xshootc, args=(gammac2, Epc, kc))
#         yfc = -1 * np.sqrt(16*(kc**2)-Epc) * phic[-1,0] # Boundary value
#         normc = np.trapz(phic[:, 0] * phic[:, 0], xshootc) # normalization integral
#         if abs(phic[-1, 1] - yfc) < tolc and abs(normc - 1) < tolc:  # check for convergence of boundary condition and normalization
#             print('Ep = '+ str(Epc))  # write out eigenvalue
#             print('A = ' + str(Ac))
#             break  # get out of convergence loop
#         else:
#             if abs(normc - 1) > tolc:
#                 Ac = Ac/np.sqrt(normc)
#             if abs(phic[-1, 1] - yfc) > tolc:
#                 if (-1) ** (modec + 1) * (phic[-1, 1] - yfc) > 0:
#                     Epc += dEpc
#                 else:
#                     Epc -= dEpc / 2
#                     dEpc /= 2

#     print('converged!')
#     eigenvaluesc2[modec-1,] = Epc
#     Ep_startc = Epc + 0.1  # after finding eigenvalue, pick new start
#     normc = np.trapz(phic[:, 0] * phic[:, 0], xshootc)  # calculate the normalization
#     print('Norm = ' + str(normc))
#     eigenfunctionsc2[:, modec-1] = np.abs(phic[:,0] / np.sqrt(normc))
    
#     plt.plot(xshootc, phic[:, 0] / np.sqrt(normc), colc[modec - 1])  # plot modes
    
    
# plt.show()  # end mode loop

# A7 = eigenfunctionsc2
# A8 = eigenvaluesc2

# print(eigenfunctionsc2)
# print(eigenvaluesc2)
#######
# d
print('running part d')
Epd = 1
gammad = 0
xranged = [-2, 2]
xd =  np.arange(-2, 2, 0.1)
kd = 1
phi0d = np.array([1, np.sqrt((4)-1)])
tolds = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

def rhsd(xd, yd, Epd):
    return [yd[1], (np.square(xd) - Epd) * yd[0]]

rk45_steps = []
rk23_steps = []
radau_steps = []
bdf_steps = []

for told in tolds:
    options = {'rtol': told, 'atol': told}
    rk45_sol = solve_ivp(rhsd, xranged, phi0d, method='RK45', args=(Epd,), **options)
    rk45_step = np.mean(np.diff(rk45_sol.t))
    rk23_sol = solve_ivp(rhsd, xranged, phi0d, method='RK23', args=(Epd,), **options)
    rk23_step = np.mean(np.diff(rk23_sol.t))
    radau_sol = solve_ivp(rhsd, xranged, phi0d, method='Radau', args=(Epd,), **options)
    radau_step = np.mean(np.diff(radau_sol.t))
    bdf_sol = solve_ivp(rhsd, xranged, phi0d, method='BDF', args=(Epd,), **options)
    bdf_step = np.mean(np.diff(bdf_sol.t))
    
    rk45_steps.append(rk45_step)
    rk23_steps.append(rk23_step)
    radau_steps.append(radau_step)
    bdf_steps.append(bdf_step)

rk45_slope = np.polyfit(np.log(rk45_steps), np.log(tolds), 1)[0]
rk23_slope = np.polyfit(np.log(rk23_steps), np.log(tolds), 1)[0]
radau_slope = np.polyfit(np.log(radau_steps), np.log(tolds), 1)[0]
bdf_slope = np.polyfit(np.log(bdf_steps), np.log(tolds), 1)[0]

slopes = np.array([[rk45_slope], [rk23_slope], [radau_slope], [bdf_slope]])
A9 = slopes.flatten()
print(np.shape(A9))

plt.plot(np.log(rk45_steps), np.log(tolds))
plt.plot(np.log(rk23_steps), np.log(tolds))
plt.plot(np.log(radau_steps), np.log(tolds))
plt.plot(np.log(bdf_steps), np.log(tolds))
plt.show()
#######
# e
print('running part e')
eigenvalues_exact = 2*np.arange(0,5) + (1)

def hermite(xi, n):
    if n == 0: return 1
    if n == 1: return 2*xi
    return 2*xi*hermite(xi, n-1) - 2*(n-1)*hermite(xi, n-2)
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
def psi(xi, n, hermite):
    return np.pi**(-1/4) / np.sqrt(2**n * factorial(n)) * hermite * np.exp(-xi**2 / 2) 

xe = np.linspace(-4, 4, 81)
eigenfunction1 = np.abs(psi(xe, 0, hermite(xe, 0)))
eigenfunction2 = np.abs(psi(xe, 1, hermite(xe, 1)))
eigenfunction3 = np.abs(psi(xe, 2, hermite(xe, 2)))
eigenfunction4 = np.abs(psi(xe, 3, hermite(xe, 3)))
eigenfunction5 = np.abs(psi(xe, 4, hermite(xe, 4)))
eigenfunctions_exact = np.transpose(np.vstack((eigenfunction1, eigenfunction2, eigenfunction3, eigenfunction4, eigenfunction5)))

eigenfunctions_errora = np.abs(A1)
eigenfunctions_errorb = np.abs(A3)
for modef in range(0, 5):
        eigenfunctions_errora[:, modef] -= eigenfunctions_exact[:, modef]
        eigenfunctions_errorb[:, modef] -= eigenfunctions_exact[:, modef]
        
eigenfunctions_errora = np.trapz(np.square(eigenfunctions_errora), xe,  axis = 0)
eigenfunctions_errorb = np.trapz(np.square(eigenfunctions_errorb), xe, axis = 0)

Ep_errora = 100*(np.abs(eigenvalues_exact-A2)/eigenvalues_exact)
print(Ep_errora)
Ep_errorb = 100*(np.abs(eigenvalues_exact-A4)/eigenvalues_exact)

A10 = eigenfunctions_errora
A11 = Ep_errora
A12 = eigenfunctions_errorb
A13 = Ep_errorb
print(np.shape(A10))
print(np.shape(A11))
print(np.shape(A12))
print(np.shape(A13))
print(A10)
print(A11)
print(A12)
print(A13)