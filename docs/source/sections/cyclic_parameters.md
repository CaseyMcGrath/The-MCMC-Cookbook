# Cyclic Parameters

Sometimes we might have a model with a parameter that has a natural periodic cycle.  The one I encounter most frequently and usually think of as an example is a phase parameter $\phi_0$ inside of a sinusoidal function $\sin(\phi_0)$.  Because sine/cosine repeat themselves every $2\pi$ radians, they are naturally cyclic on the interval $\left[0, 2\pi\right)$.

**Problem 1**

Think of what will happen in our MCMC.  Imagine we allow $\phi_0$ to take on any value between $\left(-\infty, \infty\right)$, and that the "true" value of this parameter is $\phi_0 = 4$.  Our MCMC algorithm will begin exploring the parameter space, and at some point might come close to the value $\phi_0 = 4$ and start to show that there exists a posterior mode in the parameter space at that value.  But the MCMC algorithm might continue exploring and eventually make its way to $\phi_0 = 4 + 2\pi$, or $\phi_0 = 4 - 2\pi$, or $\phi_0 = 4 + 16\pi$, and so on and so on.  Because of the nature of the cyclic interval, any value of $\phi_0 = 4 + 2\pi N$ for any integer $N$ will be a valid solution.

The problem with this is that we will end up with a runaway MCMC on this parameter because it will have an infinite number of posterior modes!  It might never do a good job of exploring the local parameter space around $\phi_0 = 4$, because if it proposes too large of a jump it might start finding support at another mode.  And if we have a model with multiple parameters, if one of them is constantly and wildly changing all of the time as the algorithm runs, it might seriously inhibit the ability of the other parameters to explore their parts of parameter space (especially if any of the parameters are covariant with $\phi_0$!).

**Problem 2**

Ok, so we could instead restrict the range of the cyclic parameter to its interval - in this example, $0 \leq \phi_0 < 2\pi$.  But imagine if the "true" value of the parameter $\phi_0$ is near one of the boundaries, either $0$ or $2\pi$.  Now as our MCMC algorithm explores the parameter space, it might begin to settle on the posterior mode near the boundary, but it will also probably really struggle to adequately explore the values at the boundary itself.

Imagine if we are at the value $\phi_0 = 6.27$.  From this value of $\phi_0$, the posterior might easily find more support crossing the cyclic boundary at $\phi_0=0.01$.  But depending on what kind of jump proposal we are using, it will probably be *very unlikely* that the MCMC algorithm will be able to make this proposed jump if the boundary explicitly cuts off at $2\pi$.


````{admonition} Treating Cyclic Parameters

In order to get the appropriate behavior that we want for our MCMC algorithm to accept cyclic parameters and explore them efficiently, we will:

1. Restrict the cyclic parameters to their natural periodic interval.  We will explicitly control this in the [prior *dictionary*](./building_priors/building_priors.md#prior-dictionary) that we have been using in our examples so far.

2. Modify the [prior *function*](./building_priors/building_priors.md#prior-function) to wrap any cyclic parameter about it's interval.  This can be easily accomplished using the *modulo* operator.  This will look something like:

```
def ln_prior(param):

    ..., phi0, ... = param

    # Cyclic Parameters
    phi0 %= (2*np.pi)

    # Calculate the PDF value of the input parameter
    ...
    prior_phi0 = prior['phi0'].pdf(phi0)
    ...
```


3. Modify the MCMC algorithm in its final step to also wrap any cyclic parameters (again using the *modulo* operator) that accept proposed moves which would move them outside of their periodic interval.  This will look something like:

```
# Heart of the MCMC Algorithm: the acceptance criteria
# --> (Pseudo-Code Step 5)
if (lnprior_proposed - lnprior_current) + (lnlike_proposed - lnlike_current) + (lnjump_current - lnjump_proposed) > lnU:
    # accept the proposed sample
    x_samples[i,:] = x_proposed
    # Cyclic Parameters
    x_samples[i,1] %= (2*np.pi)
```

A benefit of doing it this way, is that we don't have to modify our jump proposals themselves to appropriately calculate forward and reverse jumps when they encounter cyclic boundaries.  We will allow the jump proposals to have the freedom to propose any potential jump possible, even ones that cross the cyclic boundary.  This way, the forward and reverse jump PDF values will be calculated at the cyclic boundaries correctly.  But then before moving on to the next MCMC iteration, we will wrap the parameter value back inside of it's appropriate interval, so that it isn't starting the subsequent iteration *outside* of the prior boundary.
````
