# MCMC 2

Let's practice making an MCMC for a model that contains a **cyclic parameter**!

## Ingredients

Here are the ingredients that we are going to use in this MCMC:

````{tab-set}

```{tab-item} MCMC Techniques
- [x] standard MCMC
- [ ] parallel tempering
- [ ] rejection sampling
```

```{tab-item} Conveniences
- [x] progress tracking bar
- [x] efficiency tracking diagnostics
    - [x] in-model jump acceptance ratios
    - [ ] temperature swap acceptance ratios
- [x] cyclic parameters
```

```{tab-item} Jump Proposals
- [x] symmetric jumps
    - [x] Gaussian/Multivariate Normal
- [ ] prior jumps
- [ ] block (gibbs) jumps
- [ ] multiple jump schemes
```

````

## The Wave

We will start with [The Wave problem](../toy_problems.md#the-wave).  We will copy/paste all of the same code we used before for [The Bump problem](../mcmc_1/mcmc_1.md#3d-bump), but [modify it appropriately](../cyclic_parameters.md) to handle the model's cyclic parameter!


```python
from tqdm import tqdm  # progress bar
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from chainconsumer import ChainConsumer, Chain, Truth
```

### Generate the Dataset

First we need to create a dataset that we will perform our Bayesian analysis on!  We will:
1. code a function for the wave model
2. create an array of observation times
3. create a signal
4. generate the underlying noise
5. create the observed dataset


```python
# Define The Wave Model as a function of the three parameters A, phi0, and f0
def Model(A, phi0, f0, t):
    return A*np.cos(phi0 + 2*np.pi*f0*t)
```


```python
# Generate the array of times at which we observe the data
starttime = 0
endtime   = 20
Tobs      = endtime - starttime

Nt        = 200  # number of timing data
times     = np.linspace(starttime, endtime, Nt)
```

```{attention}
In order to really demonstrate the success of our method for treating cyclic parameters, we are specifically going to choose a "true" value for $\phi_0$ very close to one of its natural periodic boundaries!  What we should then observe, is that the MCMC algorithm is able to efficiently cross back and forth over the periodic boundary as it searches, without getting stuck or running off and finding an infinite number of posterior modes!
```


```python
# Create the signal

# Inject the true value of the parameters we will try to recover!
injection = [1.3, 0.4, 0.32]

# calculate the signal from the model
signal    = Model(*injection, times)
```

For the sake of replicability, let us set a random seed when generating the noise:


```python
# Generate noise
# --> this dataset has uncorrelated white noise

# Setting a random seed so that you can replicate the graphs
np.random.seed(42)

sigma_n = 2
noise   = np.random.normal(0, sigma_n, size=Nt)
```


```python
# Create the observed data
data = signal + noise
```

Let's take a quick look at what this data now looks like:


```python
fig, ax = plt.subplots(1,1,figsize=(6,4))

ax.plot(times, data,   color='gray', alpha=0.5, label='data')
ax.plot(times, signal, color='C0', label='signal')

ax.legend(), ax.set_xlabel('time', fontsize=12)
plt.show()
```


    
![png](output_13_0.png)
    


###  Prior and Likelihood

Next we need to write down our prior and likelihood that we described in [The Wave](../toy_problems.md#the-wave).

With regards to the likelihood,
> **we don't need to worry about including the normalization in our definition of the likelihood function.**

```{attention}
As discussed in [Cyclic Parameters](../cyclic_parameters.md), we will define the boundaries for the cyclic parameter in our prior dictionary, and we will modify the prior function to appropriately wrap any input for the cyclic parameter around its appropriate interval.  The interval for $\phi_0$ here is $2\pi$, so for any input of this parameter into our function, we can use the modulo operator ('$\bmod$' or '$\%$'):

$\phi_0 = \phi_0 \bmod 2\pi$
```


```python
# Define a dictionary to store the prior for our parameter

prior = {
         'A':    scipy.stats.loguniform(a=1e-1, b=1e1),        #   a < x < b
         'phi0': scipy.stats.uniform(loc=0, scale=2*np.pi),    # loc < x < loc + scale
         'f0':   scipy.stats.loguniform(a=1e-1, b=1e1),        #   a < x < b
        }
```


```python
# Now use the dictionary to construct the log-prior function

def ln_prior(param):

    A, phi0, f0 = param

    # Cyclic Parameters
    phi0 %= (2*np.pi)

    # Calculate the PDF value of the input parameter
    prior_A    = prior['A'].pdf(A)
    prior_phi0 = prior['phi0'].pdf(phi0)
    prior_f0   = prior['f0'].pdf(f0)
    
    # !!Boundary check!!
    # If the parameters land out of their boundaries, let's automatically return an effective (numerical) -inf
    if (prior_A == 0) or (prior_phi0 == 0) or (prior_f0 == 0):
        return -1e300
    # Otherwise, return the log of the prior distribution
    else:
        return np.log(prior_A * prior_phi0 * prior_f0)
```

Ok, next we define the log-likelihood function:


```python
# Define the unnormalized log-likelihood function.

def ln_like(param, data, sigma_n, times):
    M = Model(*param, times)
    return (- (data - M)**2 / (2*sigma_n**2)).sum()
```

Let's do a couple of quick sanity checks, just to basically make sure that our functions are working the way we expect:


```python
print("Quick checks:")
print(r"--> log-prior of the injection              = {0:0.4f}".format(ln_prior(injection)))
print(r"--> log-prior out of cyclic parameter range = {0:0.4e}".format(ln_prior([1.9, 2*np.pi+0.1, 2.3])))
print(r"--> log-likelihood of the injection         = {0:0.4f}".format(ln_like(injection, data, sigma_n, times)))
```

    Quick checks:
    --> log-prior of the injection              = -4.0152
    --> log-prior out of cyclic parameter range = -6.3670e+00
    --> log-likelihood of the injection         = -86.4097


Ok everything seems fine, let's move on to defining our jump PDF!

### Jump Proposal

We have three model parameters, so let's just use just the basic [Multivariate Normal jump proposal](../building_jump_proposals.md#multivariate-normal-jumps).  Remember, this type of jump proposal is a [symmetric jump proposal](../building_jump_proposals.md#symmetric-jump-proposals), which means that the forward and reverse jumps will have equal probability.  So in practice, this means that the ratio of the reverse to the forward jump found in the acceptance ratio criteria {eq}`acceptance_ratio_code` will always be identically $=1$.

```{margin}
I have picked a jump Covariance matrix here that I found seemed to work decently well (through trial and error).
```


```python
# The FORWARD jump proposal

def jump_F_MultivariateNorm(sample_current):
    # Covariance matrix that set's each parameter's jump scale
    Cov = np.array([[0.1, 0,   0,  ],
                    [0,   0.1, 0,  ],
                    [0,   0,   0.0001]])
    
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method
    sample_proposed = scipy.stats.multivariate_normal(mean=np.array(sample_current), cov=Cov).rvs()
    pdf_value       = scipy.stats.multivariate_normal(mean=np.array(sample_current), cov=Cov).pdf(sample_proposed)
    
    return sample_proposed, pdf_value


# The REVERSE jump proposal

def jump_R_MultivariateNorm(sample_current, sample_proposed):
    # standard deviation of the jump
    Cov = np.array([[0.1, 0,   0,  ],
                    [0,   0.1, 0,  ],
                    [0,   0,   0.0001]])
    
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method    
    pdf_value = scipy.stats.multivariate_normal(mean=np.array(sample_proposed), cov=Cov).pdf(sample_current)
    
    return pdf_value
```

**Sanity Check:** Let's test out our two new functions and verify that they are indeed symmetric!


```python
# Pick a starting parameter value
old_sample = [4.1, 3.783, 1.2]

# Propose a new parameter value + it's PDF value using the forward jump proposal
new_sample, PDF_forward = jump_F_MultivariateNorm(old_sample)

# Now calculate what the reverse PDF value would be if we jump from the proposed parameter back to the current parameter
PDF_reverse = jump_R_MultivariateNorm(old_sample, new_sample)

print("Current Sample  =", old_sample)
print("Proposed Sample =", new_sample)
print("PDF value of Proposed sample given Current  sample (FORWARD jump) = {0:0.4f}".format(PDF_forward))
print("PDF value of Current  sample given Proposed sample (REVERSE jump) = {0:0.4f}".format(PDF_reverse))
```

    Current Sample  = [4.1, 3.783, 1.2]
    Proposed Sample = [4.27733564 3.8961423  1.21083051]
    PDF value of Proposed sample given Current  sample (FORWARD jump) = 28.3094
    PDF value of Current  sample given Proposed sample (REVERSE jump) = 28.3094


### MCMC Algorithm

Now for the MCMC algorithm!  

```{attention}
[The last thing that we need to modify](../cyclic_parameters.md) for treating our cyclic parameter is to make sure that in the very final step of the algorithm, is to take any accepted proposed sample and modulo the parameter with its cyclic interval.  This will ensure that if the jump proposal proposed and accepted a jump to a cyclic parameter that lands outside of the prior boundary, the proposed sample is wrapped back into its periodic interval correctly!
```

We have also added in below our [dynamic counter](../tracking_in-model_jumps/tracking_in-model_jumps.md#dynamic-counter) for tracking the in-model acceptance rate!


```python
# data structure 
Nsample = 200_000   # number of samples
Ndim    = 3         # number of model dimensions

# Initialize data arrays
x_samples = np.zeros((Nsample, Ndim))

# Initialize in-model jump tracking diagnostic (dynamic counter)
# --> store 0 (jump rejected) or 1 (jump accepted)
jump_counter_inmodel = np.zeros(Nsample-1)

# Starting sample
# --> (Pseudo-Code Step 1)
x_samples[0] =  [2.8, 4.7, 0.8]
```

```{margin}
Note that our model parameter $\phi_0$ in this problem is the $N_\text{dim}$ array indexed 1 here.  This is the reason for the '1' index in `x_samples[i,1] %= ...` below.
```


```python
# LOOP: Samples
for i in tqdm(range(1,Nsample)):

    # Current sample
    x_current = x_samples[i-1,:]

    # Propose NEW sample (and calculate it's FORWARD jump PDF)
    # --> (Pseudo-Code Steps 2, 3)
    x_proposed, jump_proposed = jump_F_MultivariateNorm(x_current)

    #-------------
    # Prior Check
    #-------------
    # If proposed sample is not allowed by prior, immediately reject the proposal (saves some computation)
    lnprior_proposed = ln_prior(x_proposed)
    
    if lnprior_proposed <= -1e300:
        # keep the current sample
        x_samples[i,:] = x_current
        
    #---------------------------
    # Acceptance Ratio Criteria
    #---------------------------
    # Calculate the log-prior, log-likelihood, and log-jump PDFs for the current and proposed samples
    # --> (Pseudo-Code Step 3)
    else:
        lnprior_current = ln_prior(x_current)
        
        lnlike_proposed = ln_like(x_proposed, data, sigma_n, times)
        lnlike_current  = ln_like(x_current,  data, sigma_n, times)

        lnjump_proposed = np.log( jump_proposed )
        lnjump_current  = np.log( jump_R_MultivariateNorm(x_current, x_proposed) )
    
        # Draw random number from Uniform Dist
        # --> (Pseudo-Code Step 4)
        U   = np.random.uniform(0,1)
        lnU = np.log(U)

        # Heart of the MCMC Algorithm: the acceptance criteria
        # --> (Pseudo-Code Step 5)
        if (lnprior_proposed - lnprior_current) + (lnlike_proposed - lnlike_current) + (lnjump_current - lnjump_proposed) > lnU:
            # accept the proposed sample
            x_samples[i,:] = x_proposed
            # Cyclic Parameters
            x_samples[i,1] %= (2*np.pi)
            # update the in-model jump tracking diagnostic
            jump_counter_inmodel[i-1] = 1
        else:
            # keep the current sample
            x_samples[i,:] = x_current
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199999/199999 [01:16<00:00, 2607.54it/s]



```python
# Calculate the in-model jump acceptance ratio (dynamic)
jump_acceptance_ratio_inmodel = np.cumsum(jump_counter_inmodel) / np.arange(1,Nsample,1)
```

### Result Plots

Now let's take a look at our results, to see how well our MCMC sampler worked!


```python
# store the parameter label for reference in the plots below (just for convenience)
label = [r'$A$', r'$\phi_0$', r'$f_0$']
```


```python
#--------------
# Burn-in Plot
#--------------

fig, ax = plt.subplots(Ndim,1,figsize=(12,2*Ndim), sharex=True)
plt.subplots_adjust(hspace=0.05)

for i in range(Ndim):
    ax[i].scatter(np.arange(0,Nsample,1), x_samples[:,i], s=0.5)
    ax[i].axhline(injection[i], color='k', linestyle='--')
    ax[i].set_ylabel(label[i], fontsize=12)

ax[-1].set_xlabel('Iteration', fontsize=12)
ax[0].set_title('Burn-in')

plt.show()
```


    
![png](output_33_0.png)
    



```python
#-----------------------
# Jump Acceptance Ratio
#-----------------------

fig, ax = plt.subplots(1,1,figsize=(12,2), sharex=True)
plt.subplots_adjust(hspace=0.05)

ax.scatter(np.arange(1,Nsample,1), jump_acceptance_ratio_inmodel, s=0.5)
ax.set_ylabel('In-Model Jump\nAcceptance Ratio', fontsize=12)
ax.text(0.86, 0.82, 'Average = {0:0.2f}'.format(jump_acceptance_ratio_inmodel.mean()), transform=ax.transAxes, bbox=dict(color='white',ec='k'));

ax.set_xlabel('Iteration', fontsize=12)
ax.set_title('Tracking Diagnostics')
ax.grid()

plt.show()
```


    
![png](output_34_0.png)
    


Now that we have an idea of how long it took our sampler to burn-in, let's throw away the initial samples and make a histogram of our final posterior from the remaining samples!


```python
# Discard (burn) samples
burn = 20_000

# Final posterior samples
# --> we will save two copies of the final samples: 
#     (1) one as a Pandas DataFrame (specifically for the Chainconsumer plot below),
#     (2) and the other as a regular array structure

# Pandas data stucture
PD_samples_final = pd.DataFrame(data    = x_samples[burn:],  # discard the burn-in samples
                                columns = label
                                )
# Regular array structure
x_samples_final = np.asarray(PD_samples_final)
```

Create the final corner plot of the posterior samples.


```python
#-------------
# Corner Plot
#-------------

c = ChainConsumer()

chain = Chain(samples = PD_samples_final,
              columns = label,
              name    = "MCMC 2",
              )

c.add_chain(chain)

c.add_truth(Truth(location=dict(zip(label, np.asarray(injection))), color='k'))

c.plotter.plot();
```


    
![png](output_38_0.png)
    


Let's also look examples of the MCMC inferences.


```python
#-----------------
# Inferences Plot
#-----------------

fig, ax = plt.subplots(1,1,figsize=(8,5))

ax.plot(times, data,   color='gray', alpha=0.5, label='data')

# Randomly select a subset of parameter samples
nselect = 50
indices = np.random.randint(len(x_samples_final), size=nselect)
# Now feed those parameters back into the model and see how they look plotted on our data
for ind in indices:
    model = Model(*x_samples_final[ind,:], times)
    ax.plot(times, model, color='r', alpha=2/nselect)

ax.plot(times, signal, color='C0', label='signal')

# Manually add the 'MCMC inferences' line to the legend
handles, labels = ax.get_legend_handles_labels()
line = Line2D([0], [0], label='MCMC inferences', color='r')
handles.extend([line])

ax.legend(handles=handles), ax.set_xlabel('time', fontsize=12)
plt.show()
```


    
![png](output_40_0.png)
    


I think these plots show our sampler did pretty good!  The burn-in plot specifically really gives us an idea of how well our modifications allowed the algorithm to handle the cyclic $\phi_0$ parameter!  Since we intentionally injected the true value of $\phi_0$ close to one of the boundaries, we see that the MCMC algorithm was able to easily propose and make jumps that crossed the periodic boundary.
