# The Importance of Normalizing PDFs

**Probability Density Functions** (PDFs) for continuous variables, or **Probability Mass Functions** (PMFs) for discrete variables, are normalized - integrating the PDFs (or summing the PMFs) over their variable's domain equals 1.  The posterior, prior, likelihood, and jump PDFs should all be normalized.  But if you've ever had any previous experience working with MCMC tools, you may have implemented samplers with *unnormalized likelihoods*!  So what is the deal?

Think of what is happening in the context of the heart of our MCMC sampler equation {eq}`acceptance_ratio_code`.  At the end of the day, we are calculating *ratios* of each of our PDFs, evaluated at the current and the proposed sample values.  If the normalization factor in a given PDF *does not change* for the PDF evaluated at the current or the proposed value, then it really doesn't matter if you include it because it will just cancel out in the ratio!

So if you have read a paper or perhaps a tutorial and noticed that the likelihood function is not normalized, this is probably the reason why.  But you should verify that this is the reason!  Check to make sure that whatever your likelihood function is, the normalization factor does not matter in the calculation of the ratio because it would just cancel out.

```{hint}
**Example**

Consider the likelihood function we define for The Bump problem in equation {eq}`time_domain_likelihood`.  It's overall normalization factor is:

$$
\mathcal{N} = \frac{1}{\sqrt{\left(2\pi \sigma_n^2\right)^{N_t}}} .
$$

But this normalization has no functional dependence on the value of the model parameters $\vec{x}$ in $\vec{M}\left(t,\vec{x}\right)$.  This quantity only depends on how many timing data points we have $N_t$ and the timing uncertainty in the noise $\sigma_n$.  So unless we have a separate, specific reason to calculate the properly normalized likelihood function, our MCMC algorithm itself only depends on the ratio: 

$$
\frac{ \text{like}\left(\vec{d} | \vec{x}_{i+1} \right) }{ \text{like}\left(\vec{d} | \vec{x}_{i} \right) } ,
$$

and this value will not change whether we code up the normalized or the unnormalized likelihood PDF!
```

Another important concept to understand is that likelihood PDFs are normalized over their data, while priors and jump PDFs are normalized over their parameters!  Remember that in the notation of PDFs, $\left(a | b\right)$ is "a given b" - "b" is fixed, while "a" is the thing that is changing and thus the thing that is being normalized.  In the likelihood, $\text{like}\left(\mathbf{\vec{d}}|\vec{x}\right)$, it's the data $\vec{d}$.  In the prior, $\text{pr}\left(\mathbf{\vec{x}}\right)$, it's the model parameters $\vec{x}$.  And in the jump proposals, $\text{jump}\left(\mathbf{\vec{a}}|\vec{b}\right)$, it's also the model parameters (because $\vec{a}$ here is either the current or proposed parameters $\vec{x}$).

## When it matters!

However, this is not to say that the normalizations never matter.  Because they absolutely do matter in some scenarios!  For example, normalizations are critical when building **trans-dimensional MCMC** (TDMCMC) samplers.  In TDMCMC, we are running an MCMC that jumps between different models with potentially different likelihood PDFs, and/or different numbers of parameters, from the current to the proposed state!

So if we ever are making an MCMC where we have to make a jump such that the normalization factor for the prior, likelihood, or jump *changes* in the ratio of the current to the proposed parameter value, then we must include them otherwise our MCMC will be calculating the wrong thing!
