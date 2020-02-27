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
print(alpha)

posterior_A = st.beta(alpha+conversion_A,beta+visit_A-conversion_A)
posterior_B = st.beta(alpha+conversion_B,beta+visit_B-conversion_B)
posterior_samples_A = st.beta(alpha+conversion_A,beta+visit_A-conversion_A).rvs(n_samples)
posterior_samples_B = st.beta(alpha+conversion_B,beta+visit_B-conversion_B).rvs(n_samples)

# posterior mean 
print((posterior_samples_A > posterior_samples_B).mean())

figsize(12.5,4)

#------------------------------------------------------------------
# Posterior Dist of A and B
x = np.linspace(0,1,500)

plt.plot(x, posterior_A.pdf(x), label = "posterior of A: Beta(121,1181)")
plt.plot(x, posterior_B.pdf(x), label = "posterior of B: Beta(126,1151)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Posterior Distributions of the Conversion Rates of Web Pages $A$ and $B$")
plt.legend()

#------------------------------------------------------------------
# Posterior Dist of A and B
x = np.linspace(0,0.2,500)

plt.plot(x, posterior_A.pdf(x), label = "posterior of A: Beta(121,1181)")
plt.plot(x, posterior_B.pdf(x), label = "posterior of B: Beta(126,1151)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Posterior Distributions of the Conversion Rates of Web Pages $A$ and $B$")
plt.legend()



