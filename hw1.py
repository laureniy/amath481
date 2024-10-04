import numpy as np

'''
1. Consider the function

f(x) = x sin(3x) − exp(x)

and solve for the x-value near x ≈ −0.5 that satisfies f(x) = 0. In the first part, use
the Newton-Raphson method with the initial guess x(1) = −1.6 to converge (in absolute
value) to the solution to 10−6

Keep track of the number of iterations until convergence is achieved
(NOTE: please check convergence with f(xn) not f(xn+1)). In the second part,
use bisection with the initial end points x = −0.7 and x = −0.4. Keep track of the
mid point values and number of iterations until an accuracy of 10−6 is achieved.
'''
f1 = lambda x: x*np.sin(3*x) - np.exp(x)
df1 = lambda x: np.sin(3*x) + 3*x*np.cos(3*x) - np.exp(x)
x01 = -1.6
a01 = -0.7
b01 = -0.4
ep1 = 10**(-6)

def newton(f, df, x0, ep):
    n = 0
    xns = [x0]
    xn = x0  
    for i in range(100):
        fxn = f(xn)
        dfxn = df(xn)
        n += 1
        xn = xn - fxn/dfxn
        xns.append(xn)
        if abs(fxn) <= ep:
            break
    return np.array(xns), n

A1, a3_1 = newton(f1, df1, x01, ep1)
print(A1)

def bisection(f, a0, b0, ep):
    n = 1
    mp = (a0+b0)/2
    mps = [mp]
    a = a0
    b = b0
    fa = f(a0)
    fb = f(b0)
    fmp = f(mp)
    while abs(fmp) >= ep:
        if np.sign(fmp) == np.sign(fa):
            a = mp
        else:
            b = mp
        mp = (a+b)/2
        mps.append(mp)
        fmp = f(mp)
        n += 1
    return np.array(mps), n


A2, a3_2  = bisection(f1, a01, b01, ep1)

A3 = np.array([a3_1, a3_2])

print(A2)
print(A3)
        
'''
A = ((1 2), (-1 1)) B = ((2 0), (0 2)) C = ((2 0 -3), (0 0 -1)) D = ((1 2), (2 3) (-1 0))
x = (0, 1) y = (1, 0) z = (1, 2, -1)
Calculate the following:
(a) A+B, (b) 3x - 4y, (c) Ax, (d) B(x-y), (e) Dx, (f) D y + z, (g) AB, (h) BC, (i) CD
'''
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

print(x)
print(y)
A4 = A+B
A5 = (3*x - 4*y).ravel()
A6 = (A.dot(x)).ravel()
A7 = (B.dot(x-y)).ravel()
A8 = (D.dot(x)).ravel()
A9 = (D.dot(y) + z).ravel()
A10 = np.matmul(A, B)
A11 = np.matmul(B, C)
A12 = np.matmul(C, D)

print(A4)
print(A5)
print(A6)
print(A7)
print(A8)
print(A9)
print(A10)
print(A11)
print(A12)