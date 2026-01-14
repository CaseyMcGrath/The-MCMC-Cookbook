# MCMC 1

Alright, we finally have enough background to cook up our first MCMC!


## Ingredients

Here are the ingredients that we are going to use in this MCMC:

````{tab-set}

```{tab-item} Conveniences
- [x] progress tracking bar
- [ ] efficiency tracking diagnostics
    - [ ] jump acceptance ratios
    - [ ] temperature swap acceptance ratios
- [ ] cyclic parameters
```

```{tab-item} Priors
- [x] Uniform
- [ ] Log-Uniform
- [ ] Normal
```

```{tab-item} Jump Proposals
- [x] symmetric jumps
    - [ ] Gaussian/Multivariate Normal
- [ ] prior jumps
- [ ] block (gibbs) jumps
- [ ] multiple jump schemes
```

```{tab-item} MCMC Techniques
- [x] standard MCMC
- [ ] parallel tempering
- [ ] rejection sampling
```

````

## 1D Gaussian Bump

We will start with [The Gaussian Bump problem](../toy_problems.md#the-gaussian-bump).  And we will even start by making it a model of only one parameter $t_0$, by fixing the other two parameters $A=2$ and $\sigma = 0.5$!  So the question we are essentially asking is, 

> "In the dataset I have observed, at what time $t_0$ do I think I observe a 'bump' in the data?"


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
1. code a function for the bump model
2. create an array of observation times
3. create a signal
4. generate the underlying noise
5. create the observed dataset


```python
# Define The Bump Model as a function of a single parameter: t0
def Model(t0, t):
    A     = 2
    sigma = 0.5
    return A*np.exp(-(t-t0)**2/(2*sigma**2))
```


```python
# Generate the array of times at which we observe the data
starttime = 0
endtime   = 20
Tobs      = endtime - starttime

Nt        = 200  # number of timing data
times     = np.linspace(starttime, endtime, Nt)
```


```python
# Create the signal

# Inject the true value of the parameter we will try to recover!
injection = 7.6

# calculate the signal from the model
signal    = Model(injection, times)
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


    
![png](output_12_0.png)
    


###  Prior and Likelihood

Next we need to write down our prior and likelihood.

For the prior, we will stick to the same way we did it in [Building Prior Distributions](../building_priors/building_priors.md).  Since we are trying to find the central time parameter of the bump $t_0$, a reasonable prior is a **uniform prior** - i.e. any value inside of our defined `starttime` and `endtime` are equally valid.

For the likelihood, we need to code up the likelihood function described in "[The Gaussian Bump](../toy_problems.md#the-gaussian-bump).  The normalization constant $\mathcal{N}$ does depend on our model parameter values at all (i.e., it has no functional dependence on $t_0$).  That means the value of $\mathcal{N}$ will remain constant every time we calculate the terms in equation {eq}`acceptance_ratio_code`, therefore they will always cancel out.

> **So we don't need to worry about including the normalization in our definition of the likelihood function here.**


```python
# Define a dictionary to store the prior for our parameter

prior = {
         't0': scipy.stats.uniform(loc=starttime, scale=Tobs),    # loc < x < loc + scale
     }
```


```python
# Now use the dictionary to construct the log-prior function

def ln_prior(param):

    # Calculate the PDF value of the input parameter
    prior_t0 = prior['t0'].pdf(param)

    # !!Boundary check!!
    # If the parameter lands out of its boundary, let's automatically return an effective (numerical) -inf
    if prior_t0 == 0:
        return -1e300
    # Otherwise, return the log of the prior distribution
    else:
        return np.log(prior_t0)
```

Ok, next we define the log-likelihood function:


```python
# Define the unnormalized log-likelihood function.

def ln_like(param, data, sigma_n, times):
    M = Model(param, times)
    return (- (data - M)**2 / (2*sigma_n**2)).sum()
```

Let's do a couple of quick sanity checks, just to basically make sure that our functions are working the way we expect:


```python
print("Quick checks:")
print(r"--> log-prior of the injection      = {0:0.4f}".format(ln_prior(injection)))
print(r"--> log-prior out of prior range    = {0:0.4e}".format(ln_prior(endtime+0.0001)))
print(r"--> log-likelihood of the injection = {0:0.4f}".format(ln_like(injection, data, sigma_n, times)))
```

    Quick checks:
    --> log-prior of the injection      = -2.9957
    --> log-prior out of prior range    = -1.0000e+300
    --> log-likelihood of the injection = -86.4097


Ok everything seems fine, let's move on to defining our jump PDF!

### Jump Proposal

We only have one model parameter, and we are trying to keep things for our first MCMC, so let's stick with just the basic [Gaussian jump proposal](../building_jump_proposals.md#gaussian-jumps).  Remember, this type of jump proposal is a [symmetric jump proposal](../building_jump_proposals.md#symmetric-jump-proposals), which means that the forward and reverse jumps will have equal probability.  So in practice, this means that the ratio of the reverse to the forward jump found in the acceptance ratio criteria {eq}`acceptance_ratio_code` will always be identically $=1$.

To get in the good habit while learning, let's still code everything up!  It will make things easier going forward as we move on to more complicated jump proposals, where they won't always be symmetric.  So let's start by copying over the code we already wrote for our [Gaussian jump](../building_jump_proposals.md#gaussian-jumps):


```python
# The FORWARD jump proposal

def jump_F_Gaussian(sample_current):
    # standard deviation of the jump
    std = 1
    
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method
    sample_proposed = scipy.stats.norm(loc=sample_current, scale=std).rvs()
    pdf_value       = scipy.stats.norm(loc=sample_current, scale=std).pdf(sample_proposed)
    
    return sample_proposed, pdf_value


# The REVERSE jump proposal

def jump_R_Gaussian(sample_current, sample_proposed):
    # standard deviation of the jump
    std = 1
    
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method    
    pdf_value = scipy.stats.norm(loc=sample_proposed, scale=std).pdf(sample_current)
    
    return pdf_value
```

**Sanity Check:** Let's test out our two new functions and verify that they are indeed symmetric!


```python
# Pick a starting parameter value
old_sample = 3.783

# Propose a new parameter value + it's PDF value using the forward jump proposal
new_sample, PDF_forward = jump_F_Gaussian(old_sample)

# Now calculate what the reverse PDF value would be if we jump from the proposed parameter back to the current parameter
PDF_reverse = jump_R_Gaussian(old_sample, new_sample)

print("Current Sample  = {0:0.4f}".format(old_sample))
print("Proposed Sample = {0:0.4f}".format(new_sample))
print("PDF value of Proposed sample given Current  sample (FORWARD jump) = {0:0.4f}".format(PDF_forward))
print("PDF value of Current  sample given Proposed sample (REVERSE jump) = {0:0.4f}".format(PDF_reverse))
```

    Current Sample  = 3.7830
    Proposed Sample = 4.1408
    PDF value of Proposed sample given Current  sample (FORWARD jump) = 0.3742
    PDF value of Current  sample given Proposed sample (REVERSE jump) = 0.3742


### MCMC Algorithm

Alright, the moment is upon us!  We have everything we now need to fully construct our first MCMC algorithm!  We will follow our [pseudo-code outline](../mcmc_basics.md#pseudo-code-for-the-mcmc-algorithm) and our [schematic](../schematics/schematics.md#mcmc).

First we initialize our starting sample (in this problem, *we* are going to select the starting point), and we also define our MCMC data structure.


```python
# data structure 
Nsample = 100_000   # number of samples
Ndim    = 1        # number of model dimensions

# Initialize data arrays
x_samples  = np.zeros((Nsample, Ndim))

# Starting sample
# --> (Pseudo-Code Step 1)
x_samples[0] = 13.4
```

We can achieve the steps in our [MCMC schematic](../schematics/schematics.md#mcmc) with a **single `for` loop**.  This loop will range over the number of samples that we want to draw $N_\text{samples}$.

Note, each step in our [pseudo-code](../mcmc_basics.md#pseudo-code-for-the-mcmc-algorithm) is indicated in the code comments.

After we propose a new jump, we add one additional step: the **"prior check."**  This step is not strictly necessary, but we know that if we were to propose a sample that lies outside of our parameter space, then we should not accept the jump to that parameter (because the log-prior has a (numerical) negative infinity, so our criteria equation {eq}`ln_acceptance_ratio_code` will never accept the proposal).  This check is really more useful if you have any parameters in your model where the prior forbids certain values - such as is the case with the Uniform prior that we have here!


```python
# LOOP: Samples
for i in tqdm(range(1,Nsample)):

    # Current sample
    x_current = x_samples[i-1,:]

    # Propose NEW sample (and calculate it's FORWARD jump PDF)
    # --> (Pseudo-Code Steps 2, 3)
    x_proposed, Jump_proposed = jump_F_Gaussian(x_current)

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

        lnjump_proposed = np.log( Jump_proposed )
        lnjump_current  = np.log( jump_R_Gaussian(x_current, x_proposed) )
    
        # Draw random number from Uniform Dist
        # --> (Pseudo-Code Step 4)
        U   = np.random.uniform(0,1)
        lnU = np.log(U)

        # Heart of the MCMC Algorithm: the acceptance criteria
        # --> (Pseudo-Code Step 5)
        if (lnprior_proposed - lnprior_current) + (lnlike_proposed - lnlike_current) + (lnjump_current - lnjump_proposed) > lnU:
            # accept the proposed sample
            x_samples[i,:] = x_proposed
        else:
            # keep the current sample
            x_samples[i,:] = x_current
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 99999/99999 [01:12<00:00, 1371.77it/s]


That's all there is to it!  The entire MCMC "black box" fits in the above code block.  We just cooked up a simple MCMC!

### Result Plots

Now let's take a look at our results, to see how well our MCMC sampler worked!


```python
# store the parameter label for reference in the plots below (just for convenience)
label = [r'$t_0$']
```


```python
#--------------
# Burn-in Plot
#--------------

fig, ax = plt.subplots(1,1,figsize=(12,2), sharex=True)
plt.subplots_adjust(hspace=0.05)

ax.scatter(np.arange(0,Nsample,1), x_samples, s=0.5)
ax.axhline(injection, color='k', linestyle='--')
ax.set_ylabel(label[0], fontsize=12), ax.set_xlabel('Iteration', fontsize=12)
ax.set_title('Burn-in')

plt.show()
```


    
![png](output_32_0.png)
    


Now that we have an idea of how long it took our sampler to burn-in, let's throw away the initial samples and make a histogram of our final posterior from the remaining samples!


```python
# Discard (burn) samples
burn = 20_000

# Pandas data stucture to store the final posterior samples
pdsamples = pd.DataFrame(data    = x_samples[burn:],  # discard the burn-in samples
                         columns = label
                        )
```

Create the final corner plot of the posterior samples


```python
#-------------
# Corner Plot
#-------------

c = ChainConsumer()

chain = Chain(samples = pdsamples,
              columns = label,
              name    = "mcmc 1",
              )

c.add_chain(chain)

c.add_truth(Truth(location=dict(zip(label, np.asarray([injection]))), color='k'))

c.plotter.plot();
```


    
![png](output_36_0.png)
    


Let's also look examples of the MCMC inferences.


```python
#-----------------
# Inferences Plot
#-----------------

fig, ax = plt.subplots(1,1,figsize=(8,5))

ax.plot(times, data,   color='gray', alpha=0.5, label='data')

# Randomly select a subset of parameter samples
nselect = 40
indices = np.random.randint(np.asarray(pdsamples).size, size=nselect)
# Now feed those parameters back into the model and see how they look plotted on our data
for ind in indices:
    model = Model(pdsamples[label[0]][ind], times)
    ax.plot(times, model, color='r', alpha=2/nselect)

ax.plot(times, signal, color='C0', label='signal')

# Manually add the 'MCMC inferences' line to the legend
handles, labels = ax.get_legend_handles_labels()
line = Line2D([0], [0], label='MCMC inferences', color='r')
handles.extend([line])

ax.legend(handles=handles), ax.set_xlabel('time', fontsize=12)
plt.show()
```


    
![png](output_38_0.png)
    


Looking at these three plots, I'd say things look pretty good!  The our MCMC sampler burned-in pretty quick and even though we started our initial parameter out in the 'wrong' part of parameter space, it quickly jumped around and found the 'truth' parameter value.

It shouldn't be too surprising that the posterior samples are a little offset from the truth - this often happens.  And it is because there is noise in our data - noise will always bias things somewhat.  But in our case here, the 'truth' still falls within roughly the '$1\sigma$' range of the recovered posterior on $t_0$.

And when we plot some of the MCMC inferences, again we see there is a slight offset.  But actually, just looking at the noisy data itself, you might notice that there does appear to be a slightly larger noise fluctuation just to the left of the injected truth (for this *specific* noise realization), and that might be what is pushing the model of our bump a little over from where the truth signal lies!
