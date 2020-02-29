#%%
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from scipy import stats as st 
import numpy as np

#%%
x=np.linspace(-3,3,200)
plt.plot(x,st.t(1).pdf(x),linestyle="--",label="t(1)")
plt.plot(x,st.t(5).pdf(x),linestyle="dotted",label="t(5)")
plt.plot(x,st.norm.pdf(x), label="Normal(0,1)")
plt.legend()
plt.savefig("../assaeunji.github.io/images/tdist.png")
#%%
n  = 1000
x1 = 10
x2 = 46
x3 = 80
x4 = n - x1 - x2 - x3 #x4 = 864

#%%
obs = np.array([x1,x2,x3,x4])
prior = np.array([1,1,1,1])
n_samples = 1000
posterior = st.dirichlet(prior+obs).rvs(n_samples)
posterior.shape #(1000,4) 1000개의 각 가격 플랜 별 확률

print (posterior[0].round(3))

#%%
#------------------------------------------------------------------
# Posterior

for i, label in enumerate(["p1","p2","p3","p4"]):
    ax = plt.hist (posterior[:,i], bins=50, label=label,histtype="stepfilled")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Posterior Distribution of the probability of selecting different prices")
    plt.legend()

# plt.savefig('../assaeunji.github.io/images/abtest-multi.png')
#%%
#------------------------------------------------------------------
# Expected revenue
def expected_revenue(P):
    return 75*P[:,0] + 49*P[:,1] + 25*P[:,2]

posterior_ER = expected_revenue(posterior)

plt.hist(posterior_ER, bins=50, label = "Expected revenue", histtype="stepfilled")
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Posterior Distribution of the Expected Revenue")
plt.legend()

# plt.savefig("../assaeunji.github.io/images/abtest-ER.png")
#%%
#------------------------------------------------------------------
n_A = 1000
x1_A= 10
x2_A= 46
x3_A= 80
x4_A=n_A-x1_A-x2_A-x3_A

n_B = 2000
x1_B= 45
x2_B= 84
x3_B= 200
x4_B=n_B-x1_B-x2_B-x3_B
alpha_A=np.array([1+x1_A,1+x2_A,1+x3_A,1+x4_A])
alpha_B=np.array([1+x1_B,1+x2_B,1+x3_B,1+x4_B])
p_A = st.dirichlet(alpha_A).rvs(n_samples)
p_B = st.dirichlet(alpha_B).rvs(n_samples)

ER_A=expected_revenue(p_A)
ER_B=expected_revenue(p_B)
#%%
plt.hist(ER_A,label="E[R] of A",bins=50,histtype="stepfilled",alpha=.8)
plt.hist(ER_B,label="E[R] of B",bins=50,histtype="stepfilled",alpha=.8)
plt.xlabel("Value"); plt.ylabel("Density")
plt.legend(loc="best")
# plt.savefig("../assaeunji.github.io/images/abtest-multi2.png")

print((ER_B>ER_A).mean())
#%%
plt.hist(ER_B-ER_A,histtype="stepfilled",color="red",alpha=0.5,bins=50)
plt.vlines(0,0,70,linestyle='solid')
plt.xlabel("Value")
plt.ylabel("Density")
plt.ylim(0,70)
plt.title("Posterior Distribution of Difference of E[$R_B$]-E[$R_A$]")
# plt.savefig("../assaeunji.github.io/images/abtest-multi3.png")


# %%
print((ER_B-ER_A).mean().round(3)) #1.168


