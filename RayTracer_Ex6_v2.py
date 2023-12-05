import numpy as np
import matplotlib.pyplot as plt

                                         
r_0 = 6371                                          # earth radius
z_0 = 0
z_1 = 650
z_2 = 40
phi_0_deg = np.array([0, 25, 55, 85, 89.9])#, 90])  # initial angles
N_0 = 315
H_n = 7                                             # refractivity scale height
stepsize_1 = 0.1
stepsize_2 = 1

phi_0_rad = np.deg2rad(phi_0_deg)

r_1 = np.arange(r_0, r_0 + z_1, stepsize_1)        
r_2 = np.arange(r_0, r_0 + z_1, stepsize_2)

theta_euler_1 = np.array(np.zeros(phi_0_rad.size))
theta_RK4_1 = np.array(np.zeros(phi_0_rad.size))
theta_euler_2 = np.array(np.zeros(phi_0_rad.size))
theta_RK4_2 = np.array(np.zeros(phi_0_rad.size))

"""################################### functions #######################################"""
N_r_f = lambda r: np.multiply(N_0, np.exp(np.divide(-(r-r_0), H_n)))
n_r_f = lambda r: 1 + np.multiply(10**(-6), N_r_f(r))
C = np.multiply(n_r_f(r_0), np.multiply(r_0, np.sin(phi_0_rad)))
phi_r_f = lambda r: np.arcsin(np.divide(C, np.multiply(n_r_f(r), r)))
d_theta_f = lambda r: np.divide(C, np.multiply(n_r_f(r), np.multiply(np.power(r, 2), np.cos(phi_r_f(r)))))  

d_theta_0 = d_theta_f(r_0)
phi_euler_1 = phi_0_rad
phi_euler_2 = phi_0_rad

"""################################### Euler #######################################"""
for i in r_1[:-1]:
    theta_euler_1 = np.append(theta_euler_1, np.multiply(d_theta_f(i), stepsize_1) + theta_euler_1[-len(phi_0_rad):], axis=0)
    phi_euler_1 = np.append(phi_euler_1, phi_r_f(i))
theta_euler_1 = theta_euler_1.reshape((len(r_1), len(phi_0_rad)))
phi_euler_1 = phi_euler_1.reshape((len(r_1), len(phi_0_rad)))

"""################################## Euler stepsize 2 #######################################"""
for i in r_2[:-1]:
    theta_euler_2 = np.append(theta_euler_2, np.multiply(d_theta_f(i), stepsize_2) + theta_euler_2[-len(phi_0_rad):], axis=0)
    phi_euler_2 = np.append(phi_euler_2, phi_r_f(i))
theta_euler_2 = theta_euler_2.reshape((len(r_2), len(phi_0_rad)))
phi_euler_2 = phi_euler_2.reshape((len(r_2), len(phi_0_rad)))

"""################################### Runge Kutta #######################################"""
for i in r_1[:-1]:
    k_1 = np.multiply(d_theta_f(i), stepsize_1)
    k_2 = np.multiply(d_theta_f(i+0.5*stepsize_1), stepsize_1)
    k_3 = k_2
    k_4 = np.multiply(d_theta_f(i+stepsize_1), stepsize_1)
    theta_RK4_1 = np.append(theta_RK4_1, k_1/6 + k_2/3 + k_3/3 + k_4/6 + theta_RK4_1[-len(phi_0_rad):], axis=0)
theta_RK4_1 = theta_RK4_1.reshape((int(len(r_1)), len(phi_0_rad))) 

"""################################### Runge Kutta stepsize 2 #######################################"""
for i in r_2[:-1]:
    k_1 = np.multiply(d_theta_f(i), stepsize_2)
    k_2 = np.multiply(d_theta_f(i+0.5*stepsize_2), stepsize_2)
    k_3 = k_2
    k_4 = np.multiply(d_theta_f(i+stepsize_2), stepsize_2)
    theta_RK4_2 = np.append(theta_RK4_2, k_1/6 + k_2/3 + k_3/3 + k_4/6 + theta_RK4_2[-len(phi_0_rad):], axis=0)
theta_RK4_2 = theta_RK4_2.reshape((int(len(r_2)), len(phi_0_rad))) 

"""################################### horizontal distances #######################################""" 
r_0_h = np.ones(len(r_1))*r_0
height_z1 = phi_0_rad - np.arcsin(np.multiply(np.divide(r_0_h[-1], r_1[-1]), np.sin(np.pi-phi_0_rad)))
height_z2 = phi_0_rad - np.arcsin(np.multiply(np.divide(r_0_h[-1], r_1[400]), np.sin(np.pi-phi_0_rad)))

d_straight_z1 = np.round(height_z1*r_0,5)
print('straight distance, z1=650:          ', d_straight_z1, '[km]')

d_euler_z1_1 = np.round(theta_euler_1[-1]*r_0,5)
print('distance with Euler, z1=650 & h=0.1:', d_euler_z1_1, '[km]')

d_RK4_z1_1 = np.round(theta_RK4_1[-1]*r_0, 5)
print('distance with RK4,   z1=650 & h=0.1:', d_RK4_z1_1, '[km]')

d_straight_z2 = np.round(height_z2*r_0,5)
print('straight distance, z2=40:           ', d_straight_z2, '[km]')

d_euler_z2_1 = np.round(theta_euler_1[400]*r_0,5)
print('distance with Euler, z2=40  & h=0.1:', d_euler_z2_1, '[km]')

d_RK4_z2_1 = np.round(theta_RK4_1[400]*r_0,5)
print('distance with RK4,   z2=40  & h=0.1:', d_RK4_z2_1, '[km]')

d_euler_z2_2 = np.round(theta_euler_2[40]*r_0,5)
print('distance with Euler, z2=40  & h=1.0:', d_euler_z2_2, '[km]')

d_RK4_z2_2 = np.round(theta_RK4_2[40]*r_0,5)
print('distance with RK4,   z2=40  & h=1.0:', d_RK4_z2_2, '[km]')


"""################################### theta Euler 1 #######################################"""
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,7))
# ax.set_theta_direction(-1)
# ax.set_theta_offset(np.pi/2)
# for i in range(len(phi_0_rad)):
#     ax.plot(theta_euler_1[:,i], r_1, label=f'Start at {phi_0_deg[i]}')
# ax.scatter(theta_euler_1[-1], np.array(np.ones(theta_euler_1[-1].size)*r_0+10), marker='o', color=['blue', 'orange', 'green', 'red', 'purple'])
# ax.set_rlim(r_0, r_0 + z_1)
# ax.set_thetalim(-np.pi/1000, np.pi/5)
# ax.set_rorigin(0)
# # ax.xaxis.tick_top()
# # ax.xaxis.set_label_position('top')
# plt.legend(loc='upper right')
# plt.ylabel('r [km]')
# plt.xlabel(r'$\theta$ [°]')
# plt.title('beam path through the atmosphere: Euler-step, stepsize h = 0.1 [km]', weight='bold')
# plt.savefig('theta_euler_1_z1.png')
# plt.show()

"""###################################### theta RK4 1 ###################################### """
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,7))
# ax.set_theta_direction(-1)
# ax.set_theta_offset(np.pi/2)
# ax.set_rlim(r_0, r_0 + z_1)
# ax.set_thetalim(-np.pi/1000, np.pi/5)
# ax.set_rorigin(0)
# # ax.xaxis.tick_top()
# # ax.xaxis.set_label_position('top')
# for i in range(len(phi_0_rad)):
#     ax.plot(theta_RK4_1[:,i], r_1, label=f'Start at {phi_0_deg[i]}')
# ax.scatter(theta_euler_1[-1], np.array(np.ones(theta_euler_1[-1].size)*r_0+10), marker='o', color=['blue', 'orange', 'green', 'red', 'purple'])
# plt.legend(loc='upper right')
# plt.ylabel('r [km]')
# plt.xlabel(r'$\theta$ [°]')
# plt.title('beam path through the atmosphere: RK4, stepsize h = 0.1 [km]', weight='bold')
# plt.savefig('theta_RK4_1_z1.png')
# plt.show()

"""################################# Beam Path Euler 1 ########################################"""
# r_0_h1 = np.ones(len(r_1))*r_0
# alpha_euler_1 = []
# for i in range(len(phi_0_rad)):
#     r_h1 = np.sqrt(pow(r_1, 2) + pow(r_0_h1, 2) - 2 * np.multiply(r_1, np.multiply(r_0_h1, np.cos(theta_euler_1[:,i]))))
#     gamma_euler_1 = np.arcsin(np.multiply(np.divide(r_0_h, r_h1), np.sin(theta_euler_1[:,i])))
#     alpha_euler_1.append(gamma_euler_1 + theta_euler_1[:,i])
#     alpha_euler_1[-1][0] = phi_0_rad[i]
# alpha_euler_deg_1 = np.rad2deg(alpha_euler_1)

"""################################### Beam Path RK4 1 #####################################"""
# alpha_RK4_1 = []
# for i in range(len(phi_0_rad)):
#     r_h1 = np.sqrt(pow(r_1, 2) + pow(r_0_h1, 2) - 2 * np.multiply(r_1, np.multiply(r_0_h1, np.cos(theta_RK4_1[:,i]))))
#     gamma_RK4_1 = np.arcsin(np.multiply(np.divide(r_0_h, r_h1), np.sin(theta_RK4_1[:,i])))
#     alpha_RK4_1.append(gamma_RK4_1 + theta_RK4_1[:,i])
#     alpha_RK4_1[-1][0] = phi_0_rad[i]
# alpha_deg_1 = np.rad2deg(alpha_RK4_1)

"""########################## plot beam paths until z2 and stepsize1 ###############################"""
# for i in range(len(phi_0_rad)): 
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#     ax.plot(alpha_euler_1[i], r_1, label='trajectory Euler step')
#     ax.plot(alpha_RK4_1[i], r_1-r_0, label='trajectory RK4')
#     ax.plot([phi_0_rad[i], phi_0_rad[i]], [0, 650], label='trajectory without ray prop.')
#     ax.set_theta_direction(-1)
#     ax.set_theta_offset(np.pi/2)
#     ax.set_rlim(r_0, r_0+z_2)
#     ax.set_rorigin(0)
#     theta_max = max(max(alpha_euler_1[i]), max(alpha_RK4_1[i]))
#     ax.set_thetalim(phi_0_rad[i]-np.deg2rad(5), theta_max+np.deg2rad(5))
#     plt.legend(loc='lower right')
#     plt.ylabel('Orbital height in [km]')
#     plt.xlabel('Ray propagation in []')
#     plt.title(f'trajectory at start angle {np.round(np.rad2deg(phi_0_rad[i]), 1)}')
# plt.show()

################### Beam Path Euler z2 ########################################
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,7))
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
for i in range(len(phi_0_rad)):
    ax.plot(theta_euler_1[:,i], r_1, label=f'$\phi_{0} =${phi_0_deg[i]}')
ax.scatter(theta_euler_1[400], np.array(np.ones(theta_euler_1[400].size)*r_0+1), marker='o', color=['blue', 'orange', 'green', 'red', 'purple'])
ax.set_rlim(r_0, r_0 + z_2)
ax.set_thetalim(-np.pi/1000, np.pi/20)
ax.set_yticks([r_0, r_0 + z_2])
ax.set_rorigin(0)
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position('top')
plt.legend(loc='upper right')
plt.ylabel('r [km]')
plt.xlabel(r'$\theta$ [°]')
plt.title('beam path through the atmosphere: Euler-step, stepsize h = 0.1 [km]', weight='bold')
plt.savefig('theta_euler_1_z2.png')
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,7))
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
for i in range(len(phi_0_rad)):
    ax.plot(theta_euler_2[:,i], r_2, label=f'$\phi_{0} =${phi_0_deg[i]}')
ax.scatter(theta_euler_2[40], np.array(np.ones(theta_euler_2[40].size)*r_0+1), marker='o', color=['blue', 'orange', 'green', 'red', 'purple'])
ax.set_rlim(r_0 , r_0 + z_2)
ax.set_thetalim(-np.pi/1000, np.pi/15)
ax.set_yticks([r_0, r_0 + z_2])
ax.set_rorigin(0)
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position('top')
plt.legend(loc='upper right')
plt.ylabel('r [km]')
plt.xlabel(r'$\theta$ [°]')
plt.title('beam path through the atmosphere: Euler-step, stepsize h = 1 [km]', weight='bold')
plt.savefig('theta_euler_2_z2.png')
plt.show()

################### Beam Path RK4 z2 ########################################
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,7))
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
for i in range(len(phi_0_rad)):
    ax.plot(theta_RK4_1[:,i], r_1, label=f'$\phi_{0} =${phi_0_deg[i]}')
ax.scatter(theta_RK4_1[400], np.array(np.ones(theta_RK4_1[400].size)*r_0+1), marker='o', color=['blue', 'orange', 'green', 'red', 'purple'])
ax.set_rlim(r_0, r_0 + z_2)
ax.set_thetalim(-np.pi/1000, np.pi/20)
ax.set_yticks([r_0, r_0 + z_2])
ax.set_rorigin(0)
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position('top')
plt.legend(loc='upper right')
plt.ylabel('r [km]')
plt.xlabel(r'$\theta$ [°]')
plt.title('beam path through the atmosphere: RK4-step, stepsize h = 0.1 [km]', weight='bold')
plt.savefig('theta_RK4_1_z2.png')
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,7))
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
for i in range(len(phi_0_rad)):
    ax.plot(theta_RK4_2[:,i], r_2, label=f'$\phi_{0} =${phi_0_deg[i]}')
ax.scatter(theta_RK4_2[40], np.array(np.ones(theta_RK4_2[40].size)*r_0+1), marker='o', color=['blue', 'orange', 'green', 'red', 'purple'])
ax.set_rlim(r_0 , r_0 + z_2)
ax.set_thetalim(-np.pi/1000, np.pi/20)
ax.set_yticks([r_0, r_0 + z_2])
ax.set_rorigin(0)
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position('top')
plt.legend(loc='upper right')
plt.ylabel('r [km]')
plt.xlabel(r'$\theta$ [°]')
plt.title('beam path through the atmosphere: RK4-step, stepsize h = 1 [km]', weight='bold')
plt.savefig('theta_RK4_2_z2.png')
plt.show()

################################## distances ######################################
"""
straight distance, z1=650:           [ -0. | 272.36926 | 776.63899 | 2258.77855 | 2751.75536] [km]
distance with Euler, z1=650 & h=0.1: [  0. | 272.47496 | 777.28697 | 2275.81919 | 2868.50542] [km]
distance with RK4,   z1=650 & h=0.1: [  0. | 272.47049 | 777.26601 | 2275.34364 | 2831.21799] [km]
straight distance, z2=40:            [ -0. |  18.5234  |  56.41354 |  346.71387 |  701.02563] [km]
distance with Euler, z2=40  & h=0.1: [  0. |  18.52956 |  56.45916 |  353.56504 |  803.65276] [km]
distance with RK4,   z2=40  & h=0.1: [  0. |  18.52925 |  56.45746 |  353.3474  |  766.72154] [km]
distance with Euler, z2=40  & h=1.0: [  0. |  18.53237 |  56.47446 |  355.53224 | 1251.05935] [km]
distance with RK4,   z2=40  & h=1.0: [  0. |  18.52925 |  56.45746 |  353.3474  |  815.45483] [km]
"""
