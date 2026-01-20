# Building Jump Proposal Distributions

Let's talk a little about jump proposals.  As we know, the posterior, prior, and likelihood PDFs all appear in [Bayes' Theorem](./mcmc_basics.md#basic-bayesian).  But this "jump proposal" PDF is a brand new thing that got introduced as a part of the actual [MCMC algorithm itself](./mcmc_basics.md#the-acceptance-ratio).  So what is it?  Well first and foremost, it is one of the most critical components (if not *the* most critical component) in our MCMC!

```{important}
The jump proposal is so named because it is a PDF that can generate a new random proposed sample of our model's parameters.  At every iteration, our MCMC is sitting at some current value of our parameters $\vec{x}_i$, and it uses the jump proposal to propose a "jump" to a proposed new value of our parameters $\vec{x}_{i+1}$.

The jump proposal can be as simple or as complicated of a PDF as you want to define.  But often the efficiency of the MCMC sampler (i.e. how quickly it can begin returning "true" samples from the underlying posterior distribution) will largely depend on how well chosen of a jump proposal we are using.
```

```{margin}
You may read papers where for a specific study, the investigators create their own "custom" MCMC.  Often this means that they will have create some sort of specialized mixture of different jump proposals that works well for sampling their specific posterior distribution.  If you are curious to see a real-world example of this, {cite:t}`CorbinCornish_2010` describe in their Section 3.3 of their work how they built a (parallel tempered) MCMC sampler that employed, "A combination of six proposal distributions."  They then briefly describe (in words) what each jump proposal was doing and why it was useful in their cocktail!
```
Additionally, the jump proposal can really be a collection of *multiple* jump proposal PDFs.  Often times for more complicated problems, a single type of jump proposal won't be good enough for the efficiency we desire, so we will create a "cocktail" of multiple jump proposals that get used together.  More on this idea later - first, let's figure out how to define a general jump proposal!


## Forward and Reverse Jumps

Recall that earlier we said that the jump PDF needs to be able to generate new random parameter samples, and it needs to be able to be able to calculate the PDF value of a proposed sample given the current sample.  And looking again at our expression in the [acceptance ratio criteria](./mcmc_basics.md#translation-to-code-the-heart-of-the-mcmc-algorithm), we have two terms that need to be evaluated.  These are the "forward" and the "reverse" jumps:

```{margin}
Note there is a little bit of asymmetry here, in the sense that we are saying our **forward** jump needs to be able to do two things, while the **reverse** jump only needs to be able to do one.  This is for two reasons.  One is that we know from the [acceptance ratio criteria](./mcmc_basics.md#translation-to-code-the-heart-of-the-mcmc-algorithm) both of these things need to be able to calculate a PDF value.  But in addition to that, we said in the [pseudo-code](./mcmc_basics.md#pseudo-code-for-the-mcmc-algorithm) that the forward jump needs to be able to generate new random samples.  Hence we are going to give it *two* tasks!
```
```{admonition} The "Forward" and "Reverse" Jump PDFs
$\text{J}\left(\vec{x}_{i+1}|\vec{x}_{i}\right)$ is the **"forward jump proposal"**.  It is the PDF value of jumping to the proposed position of parameter space given the *current position* in parameter space.  The forward jump needs to be able to:
1. Generate a new random parameter sample.
2. Calculate the PDF value of the proposed sample given the current sample.

$\text{J}\left(\vec{x}_{i}|\vec{x}_{i+1}\right)$ is the **"reverse jump proposal"**.  It is the PDF value of jumping to the current position of parameter space given the *proposed position* in parameter space.  The reverse jump needs to be able to:
1. Calculate the PDF value of the current sample given the proposed sample.
```

## Gaussian Jumps

```{margin}
I strongly recommend that you check out {cite:t}`Ellis_2018` for a nice visual explanation of the Gaussian jump!
```
[The Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) jump proposal is perhaps the easiest and most common starting point when cooking up an MCMC.  If our model has only a *single* parameter (or if we are creating the jump for just one of our model parameters) then this is the functional form of our Gaussian PDF:

$$
\text{J}\left(a | b \right) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left[-\frac{1}{2}\frac{\left(a - b\right)^2}{\sigma^2}\right]
$$

As a function, it says, "return the value of the PDF at some value $a$, centered at the value of $b$."  So the mean of this Gaussian is $b$, with standard deviation $\sigma$.

And the nice thing is that [*SciPy* already has this distribution defined](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy-stats-norm) - we can call it as a function to calculate the PDF value at $a$ (given $b$), and we can also use it to generate new random variables.

### Example

Now let's code up two functions for our forward and reverse jump proposals, that achieve the needed aspects mentioned above.  

Here I am making a choice to have my **forward jump** return a `tuple` object.  It takes as input the current parameter value, and returns both a new proposed parameter value (randomly drawn from my Gaussian), and the PDF of the proposed sample given the current sample.

The **reverse jump** only needs to return the value of the PDF of the current sample given the proposed sample (provided we give it as input both of those values).


```python
import scipy.stats
```


```python
# The FORWARD jump proposal

def jump_F_Gaussian(sample_current):
    # standard deviation of the jump
    std = 0.3
    
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method
    sample_proposed = scipy.stats.norm(loc=sample_current, scale=std).rvs()
    pdf_value       = scipy.stats.norm(loc=sample_current, scale=std).pdf(sample_proposed)
    
    return sample_proposed, pdf_value


# The REVERSE jump proposal

def jump_R_Gaussian(sample_current, sample_proposed):
    # standard deviation of the jump
    std = 0.3
    
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method    
    pdf_value = scipy.stats.norm(loc=sample_proposed, scale=std).pdf(sample_current)
    
    return pdf_value
```

Let's test out our two new functions on a little example!  Try copying this code and run the following cell multiple times - what do you notice?


```python
# Pick a starting parameter value
old_sample = 8.36

# Propose a new parameter value + it's PDF value using the forward jump proposal
new_sample, PDF_forward = jump_F_Gaussian(old_sample)

# Now calculate what the reverse PDF value would be if we jump from the proposed parameter back to the current parameter
PDF_reverse = jump_R_Gaussian(old_sample, new_sample)

print("Current Sample  = {0:0.4f}".format(old_sample))
print("Proposed Sample = {0:0.4f}".format(new_sample))
print("PDF value of Proposed sample given Current  sample (FORWARD jump) = {0:0.4f}".format(PDF_forward))
print("PDF value of Current  sample given Proposed sample (REVERSE jump) = {0:0.4f}".format(PDF_reverse))
```

    Current Sample  = 8.3600
    Proposed Sample = 8.1677
    PDF value of Proposed sample given Current  sample (FORWARD jump) = 1.0828
    PDF value of Current  sample given Proposed sample (REVERSE jump) = 1.0828


You should notice that no matter how many times you run the cell, actual value of the PDF for the forward and reverse jumps does not change, even though the proposed sample is different each time.

## Multivariate Normal Jumps

[The Multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) jump proposal is just the generalization of the Gaussian jump proposal for multiple parameters.  If we are creating a jump proposal for *multiple* parameters with dimension $k$, then this is the functional form of our Multivariate Normal PDF:

$$
\text{J}\left(\vec{a} | \vec{b} \right) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp\left[-\frac{1}{2} \left(\vec{a}-\vec{b}\right)^T \Sigma^{-1} \left(\vec{a}-\vec{b}\right)   \right]
$$

Once again, [*SciPy* has this distribution covered](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html#scipy-stats-multivariate-normal), so we don't have to reinvent the wheel here.

### Example

Now let's code up two functions for our forward and reverse jump proposals, that achieve the needed aspects mentioned above.  We are going to follow the same general structure as we did above for the Gaussian, so this is now just a slight modification of what we have already created!


```python
import scipy.stats
import numpy as np
```


```python
# The FORWARD jump proposal

def jump_F_MultivariateNorm(sample_current):
    # Covariance matrix that set's each parameter's jump scale
    Cov = np.array([[0.3, 0    ],
                    [0,   0.5]])
    
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method
    sample_proposed = scipy.stats.multivariate_normal(mean=np.array(sample_current), cov=Cov).rvs()
    pdf_value       = scipy.stats.multivariate_normal(mean=np.array(sample_current), cov=Cov).pdf(sample_proposed)
    
    return sample_proposed, pdf_value


# The REVERSE jump proposal

def jump_R_MultivariateNorm(sample_current, sample_proposed):
    # standard deviation of the jump
    Cov = np.array([[0.3, 0    ],
                    [0,   0.5]])
    
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method    
    pdf_value = scipy.stats.multivariate_normal(mean=np.array(sample_proposed), cov=Cov).pdf(sample_current)
    
    return pdf_value
```

Let's test out our two new functions on a little example!  Try copying this code and run the following cell multiple times - what do you notice?


```python
# Pick a starting parameter value
old_sample = [8.36, -2.37]

# Propose a new parameter value + it's PDF value using the forward jump proposal
new_sample, PDF_forward = jump_F_MultivariateNorm(old_sample)

# Now calculate what the reverse PDF value would be if we jump from the proposed parameter back to the current parameter
PDF_reverse = jump_R_MultivariateNorm(old_sample, new_sample)

print("Current Sample  =", *["{0:0.4f}, ".format(param) for param in old_sample])
print("Proposed Sample =", *["{0:0.4f}, ".format(param) for param in new_sample])
print("PDF value of Proposed sample given Current  sample (FORWARD jump) = {0:0.4f}".format(PDF_forward))
print("PDF value of Current  sample given Proposed sample (REVERSE jump) = {0:0.4f}".format(PDF_reverse))
```

    Current Sample  = 8.3600,  -2.3700, 
    Proposed Sample = 9.1061,  -2.7247, 
    PDF value of Proposed sample given Current  sample (FORWARD jump) = 0.1433
    PDF value of Current  sample given Proposed sample (REVERSE jump) = 0.1433


You should notice that no matter how many times you run the cell, actual value of the PDF for the forward and reverse jumps does not change, even though the proposed sample is different each time.

## Symmetric Jump Proposals

The Gaussian / Multivariate Normal jump proposals are a special type of jump proposal known as a **"Symmetric Jump Proposals."** By definition, symmetric jumps have the same forward and reverse PDF value.  In other words, symmetric jump proposals will have the same probability of jumping from the current position to the proposed position, as they do from the proposed position to the current position.  This has an important consequence, namely:

```{important}
**Symmetric Jumps** result in the ratio of the reverse to forward jump found in the [acceptance ratio criteria](./mcmc_basics.md#translation-to-code-the-heart-of-the-mcmc-algorithm) being identically $=1$ because:

$$
\text{J}\left(\vec{x}_{i}|\vec{x}_{i+1}\right) = \text{J}\left(\vec{x}_{i+1}|\vec{x}_{i}\right) .
$$
```

Did you notice that no matter what, when you run the above cells repeatedly, even though the proposed sample is different every time, and the PDF values themselves are different, the forward and reverse PDFs always match?  This is the reason!  And moreover, you can see mathematically that $J\left(a|b\right) = J\left(b|a\right)$ for the Gaussian jump (and $J\left(\vec{a}|\vec{b}\right) = J\left(\vec{b}|\vec{a}\right)$ for the Multivariate Normal jump).

This is convenient, because if we can prove mathematically that the jump proposal we want to use for our MCMC is symmetric, then we don't really have to spend computation time calculating it in the [acceptance ratio criteria](./mcmc_basics.md#translation-to-code-the-heart-of-the-mcmc-algorithm).  However, while learning all of this I personally found it really easy to miss this point, and it later caused me confusion when trying to understand how to build symmetric and non-symmetric jumps.  So for the purpose of learning and consistency, for symmetric jump proposals in **The MCMC Cookbook** we will still explicitly write this out and calculate it in our MCMCs (even at the expense of *maybe* adding some unnecessary computation time).

## Prior Jumps

A very useful type of jump proposal to include in an MCMC is a general "prior jump."  As a function, it says, "draw a random sample from the prior PDF itself, and return the PDF value."

So this jump proposal will directly make use of [whatever prior distributions](./building_priors/building_priors.md) we are using for our model parameters.  In essense it is a very "uninformed jump."  For example, the [Gaussian](#gaussian-jumps) and [Multivariate Normal](#multivariate-normal-jumps) jumps both use the current parameter sample in the MCMC to draw the next proposed sample.  But the prior jump does not - it is like blindly throwing a dart at a dart board.  The only constraint with the prior jump is that the next proposed parameter sample must just exist somewhere within the prior space that we have defined!

### Example

Now let's code up two functions for our forward and reverse jump proposals, that achieve the needed aspects mentioned above.  We are going to follow the same general structure as we have above.

Also, since the prior jump will entirely depend on the prior PDFs we choose for our specific problem of interest, let's use the prior shown in [Building Prior Distributions](./building_priors/building_priors.md) for this specific example.  Following the structure we have already described, that means we will first choose to set up a prior dictionary to store the prior PDFs of our model's parameters:


```python
import scipy.stats
import numpy as np
```


```python
# Define a dictionary to store the priors for 3 different parameters

priors = {
          0: scipy.stats.uniform(loc=3, scale=7),    # loc < x < loc + scale
          1: scipy.stats.loguniform(a=1e-1, b=1e1),  #   a < x < b
          2: scipy.stats.norm(loc=5, scale=1),       # loc = mean, scale = standard deviation
         }
```

Then we will go ahead and use the prior dictionary to construct our forward and reverse jump proposals.  This example model has three parameters, so we draw a new parameter value from each of their respective PDFs, and then calculate the PDF value of those new parameters.

```{important}
As was discussed in the [Joint Prior Normalization section](./building_priors/building_priors.md#joint-prior-normalization), we are also assuming here that the prior we are working with has independent parameters, so the joint jump PDF just multiplies each individual PDF together.
```


```python
# The FORWARD jump proposal

def jump_F_prior(sample_current):
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method
    # NOTE: no actual functional dependence on sample_current!
    sample_proposed = np.array([priors[0].rvs(), priors[1].rvs(), priors[2].rvs()])
    pdf_value       = priors[0].pdf(sample_proposed[0]) * priors[1].pdf(sample_proposed[1]) * priors[2].pdf(sample_proposed[2])
    
    return sample_proposed, pdf_value


# The REVERSE jump proposal

def jump_R_prior(sample_current, sample_proposed):
    # draw a new random sample using the .RVS() method, and calculate the PDF value using the .PDF() method    
    # NOTE: no actual functional dependence on sample_proposed!
    pdf_value = priors[0].pdf(sample_current[0]) * priors[1].pdf(sample_current[1]) * priors[2].pdf(sample_current[2])
    
    return pdf_value
```

```{attention}
As mentioned above, the forward jump doesn't actually (functionally) depend on the current sample, and the reverse jump doesn't actually (functionally) depend on the proposed jump.  So why even bother coding the functions above such that they have them as inputs?!

The answer is - you don't have to!  I am 'future proofing' things a little here - trying to make it so that all of the jump proposals that we define here are structurally set up in the exact same way.  This will make our lives easier when we start to build our MCMC algorithm such that it uses *multiple* jump schemes, not just one!
```

Let's test out our two new functions on a little example!  Try copying this code and run the following cell multiple times - what do you notice?


```python
# Pick a starting parameter value
old_sample = [3, 0.1, 6.2]

# Propose a new parameter value + it's PDF value using the forward jump proposal
new_sample, PDF_forward = jump_F_prior(old_sample)

# Now calculate what the reverse PDF value would be if we jump from the proposed parameter back to the current parameter
PDF_reverse = jump_R_prior(old_sample, new_sample)

print("Current Sample  =", old_sample)
print("Proposed Sample =", new_sample)
print("PDF value of Proposed sample given Current  sample (FORWARD jump) = {0:0.4f}".format(PDF_forward))
print("PDF value of Current  sample given Proposed sample (REVERSE jump) = {0:0.4f}".format(PDF_reverse))
```

    Current Sample  = [3, 0.1, 6.2]
    Proposed Sample = [7.730241   0.33263233 4.57811708]
    PDF value of Proposed sample given Current  sample (FORWARD jump) = 0.0340
    PDF value of Current  sample given Proposed sample (REVERSE jump) = 0.0602


Unlike with the symmetric jump proposals, now you should notice that the forward and reverse jumps end up having different PDF values every time you generate a new proposed sample!

> The prior jump proposal is a **non-symmetric** jump proposal!

### An Interesting Observation about Prior Jumps

There is a rather interesting and helpful observation to make about prior jumps that is unique and offers a deeper insight into what we are doing here.

Let's say that we have no likelihood function, or that we intentionally choose to set our likelihood function to always return the value $\text{like}\left(\vec{d} | \vec{x}\right) = 1$ no matter what set of parameters $\vec{x}$ that we give it.  Now look at equation {eq}`acceptance_ratio`.  The ratio of the likelihood functions is now gone (it is just identically $1$ in this scenario).  If the jump PDF is just equal to the prior PDF, i.e. $\text{J}\left(\vec{a}|\vec{b}\right) = \text{pr}\left(\vec{a}\right)$, then the acceptance ratio simplifies down to only:

$$
\frac{\text{pr}\left(\vec{x}_{i+1}\right)}{\text{pr}\left(\vec{x}_{i}\right)} \ \frac{\text{J}\left(\vec{x}_{i}|\vec{x}_{i+1}\right)}{\text{J}\left(\vec{x}_{i+1}|\vec{x}_{i}\right)} \ = \ \frac{\text{pr}\left(\vec{x}_{i+1}\right)}{\text{pr}\left(\vec{x}_{i}\right)} \ \frac{\text{pr}\left(\vec{x}_{i}\right)}{\text{pr}\left(\vec{x}_{i+1}\right)} \ \equiv \ 1 .
$$

Remember that **Critical Observation** discussed in [The Acceptance Ratio](./mcmc_basics.md#the-acceptance-ratio)?!  Well, here is one of the consequences of that observation!

Conceptually what is happening here should make sense.  In this scenario we are saying that the underlying posterior PDF we are trying to sample is just our prior PDF.  So if we build and MCMC that *also* uses a prior PDF to propose jumps, then we have a *perfectly efficient sampler*, because we are using the same distribution to propose jumps as the underlying distribution we are trying to sample!  We are sampling the underlying distribution, *with the underlying distribution!*


````{tip}
Practically, this actually gives us a very useful tool for **testing** our MCMC to make sure that we can return something that we *know* our MCMC should return.  Namely, once we set-up our entire MCMC algorithm and are ready to start testing it, 

1. if we hard-code our likelihood function to always return $1$ (i.e. to effectively remove our likelihood function), 

```
def ln_like(...):
    ...
    return 1
```

2. and if we draw from a prior jump proposal 100% of the time,

then we should be able to verify that our jumps are *always* accepted - i.e. [the jump acceptance ratio](./tracking_in-model_jumps/tracking_in-model_jumps.md) should return exactly $1$ for the entire MCMC simulation!  Go ahead, give it a try and see for yourself!
````
