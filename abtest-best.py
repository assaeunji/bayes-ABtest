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

nu = pm.Exponential("nu_m1",1/29)+1


obs_A = pm.NoncentralT("obs_A",mu=mu_A,lam=1/std_A**2, nu=nu, observed=True,value=duration_A)
obs_B = pm.NoncentralT("obs_B",mu=mu_B,lam=1/std_A**2, nu=nu, observed=True,value=duration_B)

mcmc = pm.MCMC([obs_A,obs_B,mu_A,mu_B,std_A,std_B,nu_m1])
mcmc.sample(25000,1000)

# %%
mu_A_trace = mcmc.trace("mu_A")[:]
mu_B_trace = mcmc.trace("mu_B")[:]
std_A_trace = mcmc.trace("std_A")[:]
std_B_trace = mcmc.trace("std_B")[:] #[:]: trace object => ndarray
nu_trace    = mcmc.trace("nu_m1")[:]+1

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
plt.savefig("../assaeunji.github.io/images/abtest-duration.png")

# %%
# Difference
print("A평균:{:.3f}, B평균:{:.3f}, A-B평균:{:.3f}".format(mu_A_trace.mean(), mu_B_trace.mean(), (mu_A_trace-mu_B_trace).mean()))


# %%
def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hpd(x, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))


# %%
# pm.hpd(mu_A_trace)
mu_A_trace1 = mu_A_trace.copy()
mu_A_trace1.ndim
# %%
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))

# %%
