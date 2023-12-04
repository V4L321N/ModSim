import numpy as np
import matplotlib.pyplot as plt

                                         
r_0 = 6371                                    # meter
phi_0g = np.array([0, 25, 55, 89.9])#, 90])      # degrees

N_0 = 315
H_n = 7                                       # refractivity scale height
steps = 0.1
steps_u = 1


### Grad to Rad and back
def G2R(x):
    x = np.array(x)
    Rad = np.multiply(x, np.pi/180)
    return Rad
def R2G(x):
    x = np.array(x)
    Grad = np.multiply(x, 180/np.pi)
    return Grad

phi_0 = G2R(phi_0g)
Theta_euler = np.array(np.zeros(phi_0.size))
Theta_RK4 = Theta_euler

Theta_euler_u = np.array(np.zeros(phi_0.size))
Theta_RK4_u = Theta_euler_u


### steps
r = np.arange(r_0, r_0 + 650, steps)        
r_u = np.arange(r_0, r_0 + 650, steps_u)

### functions
N_r_f = lambda r: np.multiply(N_0, np.exp(np.divide(-(r-r_0), H_n)))
n_r_f = lambda r: 1 + np.multiply(10**(-6), N_r_f(r))
C = np.multiply(n_r_f(r_0), np.multiply(r_0, np.sin(phi_0)))
phi_r_f = lambda r: np.arcsin(np.divide(C, np.multiply(n_r_f(r), r)))
d_Theta_f = lambda r: np.divide(C, np.multiply(n_r_f(r), np.multiply(np.power(r, 2), np.cos(phi_r_f(r)))))  

d_Theta_0 = d_Theta_f(r_0)
phi_euler = phi_0
phi_euler_u = phi_0

### Euler ###
for i in r[:-1]:
    Theta_euler = np.append(Theta_euler, np.multiply(d_Theta_f(i), steps) + Theta_euler[-len(phi_0):], axis=0)
    phi_euler = np.append(phi_euler, phi_r_f(i))
Theta_euler = Theta_euler.reshape((len(r), len(phi_0)))
phi_euler = phi_euler.reshape((len(r), len(phi_0)))

### Euler step size 2 ###
for i in r_u[:-1]:
    Theta_euler_u = np.append(Theta_euler_u, np.multiply(d_Theta_f(i), steps_u) + Theta_euler_u[-len(phi_0):], axis=0)
    phi_euler_u = np.append(phi_euler_u, phi_r_f(i))
Theta_euler_u = Theta_euler_u.reshape((len(r_u), len(phi_0)))
phi_euler_u = phi_euler_u.reshape((len(r_u), len(phi_0)))

### Runge Kutta ###
for i in r[:-1]:
    k_1 = np.multiply(d_Theta_f(i), steps)
    k_2 = np.multiply(d_Theta_f(i+0.5*steps), steps)
    k_3 = k_2
    k_4 = np.multiply(d_Theta_f(i+steps), steps)
    Theta_RK4 = np.append(Theta_RK4, k_1/6 + k_2/3 + k_3/3 + k_4/6 + Theta_RK4[-len(phi_0):], axis=0)
Theta_RK4 = Theta_RK4.reshape((int(len(r)), len(phi_0))) 

### Runge Kutta step size 2 ###
for i in r_u[:-1]:
    k_1 = np.multiply(d_Theta_f(i), steps_u)
    k_2 = np.multiply(d_Theta_f(i+0.5*steps_u), steps_u)
    k_3 = k_2
    k_4 = np.multiply(d_Theta_f(i+steps_u), steps_u)
    Theta_RK4_u = np.append(Theta_RK4_u, k_1/6 + k_2/3 + k_3/3 + k_4/6 + Theta_RK4_u[-len(phi_0):], axis=0)
Theta_RK4_u = Theta_RK4_u.reshape((int(len(r_u)), len(phi_0))) 

### Horizontal distance ### 
r_0_h = np.ones(len(r))*r_0
ga650 = phi_0 - np.arcsin(np.multiply(np.divide(r_0_h[-1], r[-1]), np.sin(np.pi-phi_0)))
ga40 = phi_0 - np.arcsin(np.multiply(np.divide(r_0_h[-1], r[400]), np.sin(np.pi-phi_0)))

d_euler_650 = np.round(Theta_euler[-1]*r_0,3)
d_RK4_650 = np.round(Theta_RK4[-1]*r_0, 3)
d_euler_40 = np.round(Theta_euler[400]*r_0,3)
d_RK4_40 = np.round(Theta_RK4[400]*r_0,3)
d_euler_40_u = np.round(Theta_euler_u[40]*r_0,3)
d_RK4_40_u = np.round(Theta_euler[40]*r_0,3)
d_streight_650 = np.round(ga650*r_0,3)
d_streight_40 = np.round(ga40*r_0,3)
print(f'Horizontal Distances:\nEuler Step 650 km: {d_euler_650} km\nRK4 650 km: {d_RK4_650} km\nEuler step 40 km: {d_euler_40} km\nRK4 40 km: {d_RK4_40} km\nEuler Step step size 2 40 km: {d_euler_40_u} km\nRK4 step size 2 40 km: {d_RK4_40_u} km\nStreight 650 km: {d_streight_650} km\nStreight 40 km: {d_streight_40} km')

### Theta Euler ###
p = 1
plt.figure(p)
p = p + 1
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#ax.set_theta_direction(-1)
#ax.set_theta_offset(np.pi/2)
for i in range(len(phi_0)):
    ax.plot(2.5*Theta_euler[:,i], r, label=f'Start at {phi_0g[i]}')
ax.set_xticks([0, G2R(10), G2R(20), G2R(30), G2R(40), G2R(50), G2R(60), G2R(70), G2R(80), G2R(90)])
#ax.set_xticklabels(['0', '4', '8', '12', '16', '20', '24', '28', '32', '36'])
ax.set_yticks(np.arange(r_0, r_0+650, 100))
#ax.set_yticklabels(['0', '100', '200', '300', '400', '500', '600'])
ax.set_rlim(r_0, r_0 + 650)
ax.set_thetalim(-np.pi/500, np.pi/2)
ax.set_rorigin(0)
plt.legend(loc='lower right')
plt.ylabel('Orbital height in [km]')
plt.xlabel('Ray propagation in []')
plt.title('Ray tracing through the atmosphere: Euler-step')
plt.show()
#plt.savefig('ThetaEuler.png')


### Theta RK4 ###
plt.figure(p)
p = p + 1
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
ax.set_rlim(r_0, r_0 + 650)
ax.set_thetalim(-np.pi/500, np.pi/2 )
for i in range(len(phi_0)):
    ax.plot(2.5*Theta_RK4[:,i], r, label=f'Start at {phi_0g[i]}')
ax.set_xticks([0, G2R(10), G2R(20), G2R(30), G2R(40), G2R(50), G2R(60), G2R(70), G2R(80), G2R(90)])
ax.set_xticklabels(['0', '4', '8', '12', '16', '20', '24', '28', '32', '36'])
ax.set_yticks(np.arange(r_0, r_0+650, 100))
ax.set_yticklabels(['0', '100', '200', '300', '400', '500', '600'])
plt.legend(loc='lower right')
plt.ylabel('Orbital height in [km]')
plt.xlabel('Ray propagation in []')
plt.title('Ray tracing through the atmosphere: RK4\nAngle Theta')
plt.show()
#plt.savefig('ThetaRK4.png')

### Phi ###
plt.figure(p)
p = p + 1
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
ax.set_rlim(0, 650)
ax.set_thetalim(-np.pi/500, np.pi/2 )
for i in range(len(phi_0)):
    ax.plot(phi_euler[:,i], r-r_0, label=f'Start at {phi_0g[i]}')
plt.legend(loc='lower right')
plt.ylabel('Orbital height in [km]')
plt.xlabel('Ray propagation in []')
plt.title('Ray tracing through the atmosphere: Angle Phi')
plt.show()
#plt.savefig('Phi.png')

### Beam Path Euler ###
r_0_h = np.ones(len(r))*r_0
alphaE = []
for i in range(len(phi_0)):
    r_h = np.sqrt(pow(r,2) + pow(r_0_h, 2) - 2 * np.multiply(r, np.multiply(r_0_h, np.cos(Theta_euler[:,i]))))
    gammaE = np.arcsin(np.multiply(np.divide(r_0_h, r_h), np.sin(Theta_euler[:,i])))
    alphaE.append(gammaE + Theta_euler[:,i])
    alphaE[-1][0] = phi_0[i]
alphaE_g = R2G(alphaE)

### Beam Path RK4
alphaR = []
for i in range(len(phi_0)):
    r_h = np.sqrt(pow(r,2) + pow(r_0_h, 2) - 2 * np.multiply(r, np.multiply(r_0_h, np.cos(Theta_RK4[:,i]))))
    gammaR = np.arcsin(np.multiply(np.divide(r_0_h, r_h), np.sin(Theta_RK4[:,i])))
    alphaR.append(gammaR + Theta_RK4[:,i])
    alphaR[-1][0] = phi_0[i]
alphaR_g = R2G(alphaR)

### Plot beam paths until 650km
for i in range(len(phi_0)): 
    plt.figure(p)
    p = p + 1
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_rlim(0, 650)
    maxtheta = max(max(alphaE[i]), max(alphaR[i]))
    ax.set_thetalim(phi_0[i]-G2R(10), maxtheta+G2R(10))
    ax.plot(alphaE[i], r-r_0, label='trajectory Euler step')
    ax.plot(alphaR[i], r-r_0, label='trajectory RK4')
    ax.plot([phi_0[i], phi_0[i]], [0, 650], label='trajectory without ray prop.')
    plt.legend(loc='lower right')
    plt.ylabel('Orbital height in [km]')
    plt.xlabel('Ray propagation in []')
    plt.title(f'trajectory at start angle {np.round(R2G(phi_0[i]), 1)}')
    #plt.savefig(f'T1_Phi{i}.png')
    plt.show()
p = p + 1

### Plot beam paths until 40km
for i in range(len(phi_0)): 
    plt.figure(p)
    p = p + 1
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    
    ax.set_rlim(0, 40)
    maxtheta = max(max(alphaE[i]), max(alphaR[i]))
    ax.set_thetalim(phi_0[i]-G2R(10), maxtheta+G2R(10))
    
    ax.plot(alphaE[i], r-r_0, label='trajectory Euler step')
    ax.plot(alphaR[i], r-r_0, label='trajectory RK4')
    ax.plot([phi_0[i], phi_0[i]], [0, 650], label='trajectory without ray prop.')
    plt.legend(loc='lower right')
    plt.ylabel('Orbital height in [km]')
    plt.xlabel('Ray propagation in []')
    plt.title(f'trajectory at start angle {np.round(R2G(phi_0[i]), 1)}')
    #plt.savefig(f'T1_Phi{i}.png')
    plt.show()
p = p + 1












