#%%
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

#%%
N=250
mu_A, std_A = 30,4
mu_B, std_B = 26,7

# %%
duration_A = np.random.normal(mu_A,std_A,size=N)
duration_B = np.random.normal(mu_B,std_B,size=N)

# %%
print (duration_B[:8])

# %%
pooled_mean = np.r_[duration_A,duration_B].mean()
pooled_std = np.r_[duration_A,duration_B].std()
tau = 1/(1000*pooled_std**2) # precision parameter
# %%
# Prior on mu_A and mu_B
with pm.Model() as model:
    mu_A = pm.Normal("mu_A",mu=pooled_mean,sd=10*pooled_std)
    mu_B = pm.Normal("mu_B",mu=pooled_mean,sd=10*pooled_std)

    std_A = pm.Uniform("std_A", pooled_std/1000, 1000*pooled_std)
    std_B = pm.Uniform("std_B", pooled_std/1000, 1000*pooled_std)

    nu = pm.Exponential("nu-1",1/29)+1

    lambda_A = std_A**-2
    lambda_B = std_B**-2

    obs_A = pm.StudentT("obs_A",mu=mu_A,lam=lambda_A, nu=nu, observed=duration_A)
    obs_B = pm.StudentT("obs_B",mu=mu_B,lam=lambda_B, nu=nu, observed=duration_B)

    # If you want to get MCMC samples of some transformations of parameters, 
    # pm.use Deterministic
    diff_of_means = pm.Deterministic('difference of means', mu_A - mu_B)
    diff_of_stds = pm.Deterministic('difference of stds', std_A - std_B)
    effect_size = pm.Deterministic('effect size',diff_of_means / np.sqrt((std_A**2 + std_B**2) / 2))

# %%
# MCMC posterior samples 
with model:
    trace = pm.sample(iter=25000,tune=1000,thin=1,cores=1) # Broken pipe

# %%
pm.plot_posterior(trace, ['mu_A','mu_B','std_A','std_B','nu-1'],color='#87ceeb');
plt.savefig("../assaeunji.github.io/images/abtest-hdi.png")
# %%
pm.plot_posterior(trace, ['difference of means','difference of stds','effect size'],ref_val=0,color='#87ceeb')
plt.savefig("../assaeunji.github.io/images/abtest-hdi-diff.png")

# %%
pm.summary(trace, varnames=['difference of means', 'difference of stds', 'effect size'])

# %%
