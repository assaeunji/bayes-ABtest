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
print (duration_B[:8])

# %%
pooled_mean = np.r_[duration_A,duration_B].mean()
pooled_std = np.r_[duration_A,duration_B].std()
tau = 1/(1000*pooled_std**2) # precision parameter
# %%
# Prior on mu_A and mu_B
mu_A = pm.Normal("mu_A",mu=pooled_mean,tau=tau)
mu_B = pm.Normal("mu_B",mu=pooled_mean,tau=tau)

std_A = pm.Uniform("std_A", pooled_std/1000, 1000*pooled_std)
std_B = pm.Uniform("std_B", pooled_std/1000, 1000*pooled_std)

nu_m1 = pm.Exponential("nu-1",1/29)

obs_A = pm.NoncentralT("obs_A",mu_A,1/std_A**2, nu_m1+1, observed=True,value=duration_A)
obs_B = pm.NoncentralT("obs_B",mu_B,1/std_B**2, nu_m1+1, observed=True,value=duration_B)

mcmc = pm.MCMC([obs_A,obs_B,mu_A,mu_B,std_A,std_B,nu_m1])
mcmc.sample(25000,10000)

# %%
mu_A_trace = mcmc.trace("mu_A")[:]
mu_B_trace = mcmc.trace("mu_B")[:]
std_A_trace = mcmc.trace("std_A")[:]
std_B_trace = mcmc.trace("std_B")[:] #[:]: trace object => ndarray
nu_trace    = mcmc.trace("nu-1")[:]+1

# %%
def _hist(data,label,**kwargs):
    return plt.hist(data,bins=40,histtype="stepfilled",alpha=.95,label=label, **kwargs)

ax = plt.subplot(3,1,1)
_hist(mu_A_trace,"A")
_hist(mu_B_trace,"B")
plt.legend ()
plt.title("Posterior distributions of $\mu$")

ax=plt.subplot (3,1,2)
_hist(std_A_trace,"A")
_hist(std_B_trace,"B")
plt.legend ()
plt.title("Posterior distributions of $\sigma$")

ax=plt.subplot (3,1,3)
_hist(nu_trace,"",color="#7A68A6")
plt.title(r"Posterior distributions of $\nu$")
plt.xlabel("Value")
plt.ylabel("Density")
plt.tight_layout()
# plt.savefig("../assaeunji.github.io/images/abtest-duration.png")

# %%
# Difference
print("A평균:{:.3f}, B평균:{:.3f}, A-B평균:{:.3f}".format(mu_A_trace.mean(), mu_B_trace.mean(), (mu_A_trace-mu_B_trace).mean()))

# %%
pm.Matplot.plot(mcmc)
mcmc.summary()