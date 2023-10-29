#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 23:30:59 2023

@author: ting
"""

import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Problem 1
L = 1
x0 = -L
y0 = 0
xf = L
yf = 0

def bisection(f, a, b, tol):
    x = (a + b) / 2
    while np.abs(b - a) >= tol:
        if np.sign(f(x)) == np.sign(f(a)):
            a = x
        else:
            b = x
        x = (a + b) / 2
    return x

def V(x):
    return -100*(math.sin(2*x) + 1)
def f(x, y, lamda_value):
    return np.array([y[1], -(-100 * (math.sin(2 * x) + 1) + lamda_value) * y[0]])

A = 1
def shoot(lambda_value):
    tspan = np.array([x0,xf])
    init_condition = np.array([y0,A])
    sol = solve_ivp(lambda x, y: f(x,y,lambda_value), tspan,init_condition)
    return sol.y[0][-1]

lamda_value = 23
sign = np.sign(shoot(lamda_value))
dlamda = 1

num_modes = 3
eigenvals = np.zeros(num_modes)
k = 0
while k < num_modes:
    lamda_next = lamda_value + dlamda
    sign_next = np.sign(shoot(lamda_next))
    if sign != sign_next:
        eigenvals[k] = bisection(shoot, lamda_value, lamda_next, 10 ** (-8))
        k = k + 1
    lamda_value = lamda_next
    sign = sign_next

for k in range(num_modes):
    print("eigenval = {}".format(eigenvals[k]))
    tspan = np.array([x0, xf])
    init_condition = np.array([y0, A])
    sol = solve_ivp(lambda x, y: f(x, y, eigenvals[0]), tspan, init_condition, t_eval=np.array([-1, 0, 1]))
    x = sol.t
    y = sol.y[0, :]
    plt.plot(x, y)
plt.show()

# A1
A1 = eigenvals[0]

# A2
A2 = y[1]

# A3
A3 = eigenvals[1]

# A4
for k in range(num_modes):
    print("eigenval = {}".format(eigenvals[k]))
    tspan = np.array([x0, xf])
    init_condition = np.array([y0, A])
    sol = solve_ivp(lambda x, y: f(x, y, eigenvals[1]), tspan, init_condition, t_eval=np.array([-1, 0, 1]))
    x = sol.t
    y = sol.y[0, :]
    plt.plot(x, y)
A4 = y[1]

# A5
A5 = eigenvals[2]

# A6
for k in range(num_modes):
    print("eigenval = {}".format(eigenvals[k]))
    tspan = np.array([x0, xf])
    init_condition = np.array([y0, A])
    sol = solve_ivp(lambda x, y: f(x, y, eigenvals[2]), tspan, init_condition, t_eval=np.array([-1, 0, 1]))
    x = sol.t
    y = sol.y[0, :]
    plt.plot(x, y)
A6 = y[1]

# Problem 2    
x0 = 1
dx0 = 0
t0 = 0
tN = 1
dt = 0.1
t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros((2, len(t)))
x[0, 0] = x0
x[1, 0] = dx0

for k in range(len(t) - 1):
    x[0, k + 1] = ((1 + (dt ** 2) / 4) * x[0, k]) / (1 - (dt ** 2) / 4) + (dt * x[1, k]) / (1 - (dt ** 2) / 4)
    x[1, k + 1] = ((1 + (dt ** 2) / 4) * x[1, k]) / (1 - (dt ** 2) / 4) + (dt * x[0, k]) / (1 - (dt ** 2) / 4)

A7 = x[0, -1]
def true_solution(t):
    return 1/2 * (np.exp(t) + np.exp(-t))
A8 = abs(true_solution(tN) - A7)


dt2 = 0.01
t = np.arange(t0, tN + dt2 / 2, dt2)
x = np.zeros((2, len(t)))
x[0, 0] = x0
x[1, 0] = dx0

for k in range(len(t) - 1):
    x[0, k + 1] = ((1 + (dt2 ** 2) / 4) * x[0, k]) / (1 - (dt2 ** 2) / 4) + (dt2 * x[1, k]) / (1 - (dt2 ** 2) / 4)
    x[1, k + 1] = ((1 + (dt2 ** 2) / 4) * x[1, k]) / (1 - (dt2 ** 2) / 4) + (dt2 * x[0, k]) / (1 - (dt2 ** 2) / 4)

A9 = x[0, -1]
A10 = abs(true_solution(tN) - A9)

# Problem 2 c-d

t = np.arange(t0, tN + dt / 2, dt)
x = np.zeros((2, len(t)))
x[0, 0] = x0
x[1, 0] = dx0

x[0, 1] = x[0, 0] + dt * x[1, 0]
x[1, 1] = x[1, 0] - dt * x[0, 0]

for k in range(len(t) - 2):
    x[0, k + 2] = x[0, k] + 2 * dt * x[1, k + 1]
    x[1, k + 2] = x[1, k] - 2 * dt * x[0, k + 1]

A11 = x[0, -1]
def true_soution(t):
    return math.cos(t)
A12 = abs(true_soution(tN) - A11)


dt2 = 0.01
t = np.arange(t0, tN + dt2 / 2, dt2)
x = np.zeros((2, len(t)))
x[0, 0] = x0
x[1, 0] = dx0

x[0, 1] = x[0, 0] + dt2 * x[1, 0]
x[1, 1] = x[1, 0] - dt2 * x[0, 0]

for k in range(len(t) - 2):
    x[0, k + 2] = x[0, k] + 2 * dt2 * x[1, k + 1]
    x[1, k + 2] = x[1, k] - 2 * dt2 * x[0, k + 1]

A13 = x[0, -1]
def true_soution(t):
    return math.cos(t)
A14 = abs(true_soution(tN) - A13)


# Problem 3-a
x0 = -0.5
y0 = -0.5
xN = 0.5
yN = 0.5
dx = 0.1

def true_sol(x):
    return x

def p(x):
    return -x/ (1 - x**2)

def q(x):
    return 1/ (1 - x**2)
    
N = 11
x = np.linspace(x0, xN, N)

A = np.zeros((N, N))
b = np.zeros((N, 1))

A[0, 0] = 1
b[0] = y0
A[N-1, N-1] = 1
b[N-1] = yN

for k in range(1, N - 1):
    A[k, k-1] = (1 - dx * p(x[k]) / 2)
    A[k, k] = (-2 + dx ** 2 * q(x[k]))
    A[k, k + 1] = (1 + dx * p(x[k]) / 2)

y = np.linalg.solve(A, b).reshape(N)
A15 = y[5]
A16 = np.max(np.abs(y - true_sol(x)))


# Problem 3-b
y0 = 0.5
yN = 0.5

def true_solu(x):
    return 1- (2*(x**2))

def q2(x):
    return 4/ (1 - x**2)
    
N = 11
x = np.linspace(x0, xN, N)

A = np.zeros((N, N))
b = np.zeros((N, 1))

A[0, 0] = 1
b[0] = y0
A[N-1, N-1] = 1
b[N-1] = yN

for k in range(1, N - 1):
    A[k, k-1] = (1 - dx * p(x[k]) / 2)
    A[k, k] = (-2 + dx ** 2 * q2(x[k]))
    A[k, k + 1] = (1 + dx * p(x[k]) / 2)

y2 = np.linalg.solve(A, b).reshape(N)
A17 = y2[5]
A18 = np.max(np.abs(y2 - true_solu(x)))

# Problem 2-c
y0 = -1/3
yN = 1/3

def true_sol(x):
    return x - (4*x**3 / 3)
def q3(x):
    return 9/ (1 - x**2)
    
N = 11
x = np.linspace(x0, xN, N)

A = np.zeros((N, N))
b = np.zeros((N, 1))

A[0, 0] = 1
b[0] = y0
A[N-1, N-1] = 1
b[N-1] = yN

for k in range(1, N - 1):
    A[k, k-1] = (1 - dx * p(x[k]) / 2)
    A[k, k] = (-2 + dx ** 2 * q3(x[k]))
    A[k, k + 1] = (1 + dx * p(x[k]) / 2)

y = np.linalg.solve(A, b).reshape(N)
A19 = y[5]
A20 = np.max(np.abs(y - true_sol(x)))