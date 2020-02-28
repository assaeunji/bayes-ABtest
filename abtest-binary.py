from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from scipy import stats as st 
import numpy as np

visit_A = 1300
visit_B = 1275

conversion_A = 120
conversion_B = 125


alpha = 1
beta  = 1
n_samples = 1000

posterior_A = st.beta(alpha+conversion_A,beta+visit_A-conversion_A)
posterior_B = st.beta(alpha+conversion_B,beta+visit_B-conversion_B)
posterior_samples_A = st.beta(alpha+conversion_A,beta+visit_A-conversion_A).rvs(n_samples)
posterior_samples_B = st.beta(alpha+conversion_B,beta+visit_B-conversion_B).rvs(n_samples)

# posterior mean 
print("{}% chance of A site better than B".format((posterior_samples_A > posterior_samples_B).mean()))

figsize(12.5,4)

#------------------------------------------------------------------
# Posterior Dist of A and B
fig,axes = plt.subplots(1,2,figsize=(10,4))
x = np.linspace(0,1,1000)
i=0
for ax in axes:
    ax.plot(x, posterior_A.pdf(x), label = "posterior of A: Beta(121,1181)")
    ax.plot(x, posterior_B.pdf(x), label = "posterior of B: Beta(126,1151)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    if i==1:
        ax.set_xlim(0.05, 0.15)
    i+=1
axes[0].legend()    

