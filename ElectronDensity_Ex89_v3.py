import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import copy

D_eff = 5e-3 # [km^2/s] # 5e3 # [m^2/s]
A0 = D_eff
A1 = A2 = A3 = 0
factor = 0.2 #1 #2
delta_b = 1#000  # [km]
b_0 = 100#000  # [m]
b_I = 600#000  # [m]
b_max = 350#000 # [m]
t_0 = 0  # [s]
t_max = 172800  # [s]
N_max = 10e11 # [m^-3]
w_N = 10#000  # [m]
N_0_0 = 2e10 # [m^-3]
N_0_i = 20e10 # [m^-3]

t_crit = delta_b**2 / (2*A0)
print(t_crit)

delta_t = factor * t_crit
b = np.arange(b_0, b_I, delta_b)
t = np.arange(t_0, t_max, delta_t)

####################### explicit solution ##################################################

# initialize N
N_e = np.zeros([len(b),len(t)])
for i in range(len(b)-1):
        N_e[i,0] = N_max * np.exp(-((b[i]-b_max)**2) / w_N**2)

# check N0/NI
N_e_bound_low = N_e[:, 0][:int(len(b)/2)] < N_0_0
N_e_bound_high = N_e[:, 0][int(len(b)/2):] < N_0_i
N_e[:int(len(b)/2),0][N_e_bound_low] = N_0_0
N_e[int(len(b)/2):,0][N_e_bound_high] = N_0_i
for i in range(len(t)):
    N_e[0,i] = N_0_0
    N_e[-1,i] = N_0_i
alpha = D_eff * delta_t / (delta_b**2)

for i  in range(len(t)-1):
    for j in range(1,len(b)-1):
        N_e[j, i+1] = N_e[j, i] + alpha * N_e[j+1, i] - 2*alpha*N_e[j, i] + alpha * N_e[j-1, i]

X, Y = np.meshgrid(b, t)
fig=plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(projection = '3d')
ax1.view_init(20, 135)
ax1.set_xlabel('b [m]', fontsize=14)
ax1.set_ylabel('t [s]', fontsize=14)
ax1.set_zlabel('N(b,t) [$m^{-3}$]', fontsize=14)
plt.title('electron density through atmosphere, ' + str(factor) + '$\Delta t_{crit}$', fontsize=16)
ax1.plot_surface(X.T,Y.T,N_e,rstride=2,cstride=10,cmap=cm.seismic)
plt.show()

####################### implicit solution ################################################

# initialize N
N_i = np.zeros([len(b),len(t)])
for i in range(len(b)-1):
        N_i[i,0] = N_max * np.exp(-((b[i]-b_max)**2) / w_N**2)

# check N0/NI
N_i_bound_low = N_i[:,0][:int(len(b)/2)] < 2e10
N_i_bound_high = N_i[:,0][int(len(b)/2):] < 20e10
N_i[:int(len(b)/2),0][N_i_bound_low] = 2e10
N_i[int(len(b)/2):,0][N_i_bound_high] = 20e10
for i in range(len(t)):
    N_i[0,i] = 2e10
    N_i[-1,i] = 20e10

alpha = D_eff * delta_t / (delta_b**2)
    
a_i = -alpha
b_i = 1 + 2*alpha
c_i = -alpha

# ########## tridiagonal matrix algorithm 
# w = np.zeros(len(b))
# g = np.zeros(len(b))
# p = np.zeros(len(b))

# for i in range(0, len(t)-1):       
#     w = copy.deepcopy(N_i[:, i])
#     g[0] = -c_i/b_i
#     p[0] = w[0]/b_i 
#     #forward Gauß
#     for j in range(1,len(b)-1):
#         denominator = b_i + a_i * g[j-1]
#         g[j] = -c_i / denominator
#         p[j] = (w[j] - a_i * p[j-1])/denominator    
#     #backward Gauß
#     for j in range(len(b)-2, 0, -1):
#         N_i[j,i+1] = g[j] * N_i[j+1,i+1] + p[j]

######### explicit matrix calculation
def N_Matrix (lower_d, main, upper_d, x) :
    n = len(x)
    w = np.zeros(n -1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)
    lower_d = lower_d * np.ones(len(x)-1)
    lower_d[-1] = 0
    main = main * np.ones(len(x))
    main[0] = 1
    main[-1] = 1
    upper_d = upper_d * np.ones (len(x)-1)
    upper_d[0] = 0
    w [0] = upper_d[0]/main[0]
    g [0] = x[0]/main[0]
    for i in range (1, n-1) :
        w[i] = upper_d[i] / (main[i] - lower_d[i-1] * w[i-1])
    for i in range (1, n):
        g[i] = (x[i] - lower_d[i-1] * g[i-1]) / (main[i] - lower_d[i-1] * w[i-1])
        p[n-1] = g[n-1]
    for i in range (n-1, 0, -1) :
        p[i-1] = g[i-1] - w[i-1]* p[i]
    return p

for i in range (1, len(t)):
    N_i[:,i] = N_Matrix(a_i, b_i, c_i, N_i[:, i-1])

X, Y = np.meshgrid(b, t)
fig=plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(projection = '3d')
ax1.view_init(elev = 20, azim = 135)
ax1.plot_surface(X.T,Y.T,N_i,rstride=2,cstride=10,cmap=cm.seismic)
ax1.set_xlim(b_0,b_I)
ax1.set_ylim(0,t_max)
ax1.set_xlabel('b [m]', fontsize=14)
ax1.set_ylabel('t [s]', fontsize=14)
ax1.set_zlabel('N(b,t) [$m^{-3}$]', fontsize=14)
plt.title('electron density through atmosphere; ' + str(factor) + '$\Delta t_{crit}$', fontsize=16)
plt.show()

####################### difference ################################################
"""
X, Y = np.meshgrid(b, t)
fig=plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(projection = '3d')
ax1.view_init(elev = 20, azim = 135)
ax1.set_xlabel('b [m]', fontsize=14)
ax1.set_ylabel('t [s]', fontsize=14)
ax1.set_zlabel('N(b,t) [$m^{-3}$]', fontsize=14)
ax1.plot_surface(X.T,Y.T,N_e - N_i,rstride=2,cstride=10,cmap=cm.seismic)
plt.title('difference of the explicit and implicit method; ' + str(factor) + '$\Delta t_{crit}$', fontsize=16)
plt.show()
"""