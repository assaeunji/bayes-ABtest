#%%
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
#%%
N=250
mu_A, std_A = 30,4
mu_B, std_B = 26,7

# %%
duration_A = np.random.normal(mu_A,std_A,size=N)
duration_B = np.random.normal(mu_B,std_B,size=N)

# %%
print (duration_A[:8])

# %%
pooled_mean = np.r_[duration_A,duration_B].mean()
pooled_std = np.r_[duration_A,duration_B].std()
tau = 1/(1000*pooled_std**2) # precision parameter
# %%
# Prior on mu_A and mu_B
mu_A = pm.Normal("mu_A",pooled_mean,tau)
mu_B = pm.Normal("mu_B",pooled_mean,tau)

# %%
std_A = pm.Uniform("std_A", pooled_std/1000, 1000*pooled_std)
std_B = pm.Uniform("std_B", pooled_std/1000, 1000*pooled_std)

# %%
nu_minus_1 = pm.Exponential("nu-1",1/29)
obs_A = pm.NoncentralT("obs_A",mu_A,1/std_A**2, nu_minus_1+1, observed=True, value = duration_A)
obs_B = pm.NoncentralT("obs_B",mu_B,1/std_B**2, nu_minus_1+1, observed=True, value = duration_B)

# %%
mcmc = pm.MCMC([obs_A,obs_B,mu_A,mu_B,std_A,std_B,nu_minus_1])
mcmc.sample(25000,10000)

# %%