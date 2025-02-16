import numpy as np
import matplotlib.pyplot as plt
from random import random as rd
from scipy.stats import chi2

'''
## MODEL 1

# data, parameters
y_obs = np.array([2.2, 2.6, 3.6, 3.7, 6.1, 7.6, 8.6, 8.8, 9.4])
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
sigma = 0.5             # sigma
alpha = 0.95     

# set up the design matrix
def design_M (x, y, s):
    # X = np.vstack((x, np.ones(len(x)))).T
    i = np.ones(x.shape)
    X = np.vstack((i, x))/s
    Y = y/s
    X = X.transpose()
    Y = Y.transpose()
    return X, Y

# set up the covariance marix
def Covariance (D_M):
    t_D_M = D_M.transpose()
    Ch = t_D_M.dot(D_M)
    C_a = np.linalg.inv(Ch)
    return C_a

# calculate matrices
A, b = design_M(x, y_obs, sigma)    
C_a = Covariance(A) 

# set up residuals
a = (C_a.dot(A.transpose())).dot(b) 
y_mod = a[1]*x + a[0]

# degrees of freedom (N - 2 for linear regression)
dof = len(x) - 2

# calculate chi squared
chi1 = sum(((y_obs - b)**2)/sigma**2)
chi_sq = ((y_obs - y_mod)/sigma)**2     
sum_chi = sum(chi_sq)
# critical value for chi squared (right tail) 
critical_chi_sq = chi2.ppf(1 - alpha, dof)
# compare calculated chi squared to the critical value
good_fit = chi_sq < critical_chi_sq
distr = chi2.pdf(y_mod, df=dof)

plt.figure(1)
plt.title('linear fit of data')
plt.ylabel('y')
plt.xlabel('x')
plt.plot(x, y_obs, 'xr', label='data')
plt.plot(x, y_mod, color='black', linestyle='dashed', label='fit')
plt.legend()

# according to table 
plt.figure(2)
plt.title('χ - distribution')
plt.xlabel('x')
plt.ylabel('densitiy')
plt.plot(y_mod, distr, color='black', label= 'χ - distribution, DoF='+str(dof))
plt.axvline(x=sum_chi, color='red', label= 'cumulative χ²')
plt.axvline(x=critical_chi_sq, color='blue', label= 'critical value for alpha=0.95')
plt.legend()
'''

## MODEL 2
#### Fitting of Excersie 2 ###########

y_obs = np.array([7.3, 3.1, 11.7, 11.9, 15.5, 31.0, 33.9, 53.0, 61.5, 71.9])
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
sigma = 4.3 
alpha = 0.95

def design_M(x, y, error):
  i = np.ones(x.shape)
  X = np.vstack((i, x))/error
  Y = y/error
  Z = x**2 / error
  X = X.transpose()
  Y = Y.transpose()
  Z = Z.transpose()
  return X, Z, Y

def covariance (D_M):
    t_D_M = D_M.transpose()
    Ch = t_D_M.dot(D_M)
    C_a = np.linalg.inv(Ch)
    return C_a

A_1, A_2, b = design_M(x, y_obs, sigma)     

A = np.column_stack([A_1, A_2])

C_a = covariance(A) 
a = (C_a.dot(A.transpose())).dot(b) 

R = np.array(np.zeros(C_a.shape))
for i in range(3):
    for j in range(3):
        R[i,j] = C_a[i,j]/np.sqrt(C_a[i,i] * C_a[j,j])

y_mod = a[0] + a[1]*x + a[2]*x**2
chi_sq = ((y_obs - y_mod)/sigma)**2 
sum_chi_sq = sum(chi_sq)
dof = len(x) - 2
critical_chi_sq = chi2.ppf(1 - alpha, dof)
distr = chi2.pdf(y_mod, df=dof)

plt.figure(3)
plt.title('quadratic fitting of data')
plt.xlabel('x')
plt.ylabel('y')
plt.errorbar(x, y_obs, yerr=2*sigma, fmt='.k')
plt.plot(x, y_obs, 'xr', label='data')
plt.plot(x, y_mod, color='black', linestyle='dashed', label='fit')
plt.legend()

plt.figure(4)
plt.title('Chi distribution with DoF='+str(dof))
plt.xlabel('x')
plt.ylabel('densitiy')
plt.plot(y_mod, distr, label='Chi distribution with DoF='+str(dof))
plt.axvline(x=sum_chi_sq, color='red', label='cumulative Chi²')
plt.axvline(x=critical_chi_sq, color='blue', label='Chi² with alpha=0.95')
plt.legend()


















plt.show()





""" first attempt
weights = 1.0 / (s**2)
# Set up the design matrix
X = np.vstack((np.ones(len(x)), x))/s
Y = (y_obs.transpose())/s
# Calculate the weighted least squares solution
A = np.dot(X.transpose(), X)
b = np.dot(Y.transpose(), Y)
solution = np.linalg.solve(A, b)
k, d = solution

y_fit = k * x + d

# Calculating the covariance matrix
variance_residuals = s**2
covariance_matrix = variance_residuals * np.linalg.inv(np.dot(X.transpose(), X))

var_k = covariance_matrix[0, 0]
var_d = covariance_matrix[1, 1]
cov_kd = covariance_matrix[0, 1]

std_k = np.sqrt(var_k)
std_d = np.sqrt(var_d)

correlation_kd = cov_kd / (std_k * std_d)

residuals = y - (k * x + d)
chi_squared = np.sum((residuals / s) ** 2)

# Degrees of freedom (N - 2 for linear regression)
dof = len(x) - 2

# Critical chi-squared value for alpha = 0.95 and dof degrees of freedom
critical_chi_squared = chi2.ppf(1 - alpha, dof)

# Compare calculated χ² to the critical value
good_fit = chi_squared < critical_chi_squared
"""
