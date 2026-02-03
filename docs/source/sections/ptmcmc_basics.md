# PTMCMC Basics

Now that we have built up our understanding of general MCMC, let's explore a slightly more advanced MCMC technique: **parallel tempering**.  Parallel tempered MCMC (or **PTMCMC**) takes everything we have already learned, and just adds one additional calculation, so it is a relatively simple extension of everything we have done so far.

First off, let's break down the terminology:

- **Parallel**: we are essentially just going to make identical copies of our MCMC algorithm, that run (mostly) independently, or "parallel," to each other.
- **Tempering**: this term is actually [rooted in the study of thermodynamics](https://en.wikipedia.org/wiki/Parallel_tempering).  Essentially every copy of our MCMC will be given a slightly different *temperature*, which modifies the behavior of the likelihood PDF.

But apart from the temperature itself, every parallel copy of our original MCMC algorithm stays the same!  This is going to make modifying our code quite easy!  We can just wrap our existing MCMC code into an additional `for` loop, and add one new calculation.

## Temperatures

Mathematically, the only thing that changes between regular MCMC and PTMCMC is our treatment of the likelihood PDF.  In PTMCMC, we make parallel copies of the MCMC algorithm, and each one calculates and explores a modified posterior distribution, where the likelihood has been raised to a power of $1/T$:

$$
\overbrace{\text{post}_T\left(\vec{x}|\vec{d}\right)}^\text{tempered posterior} = \frac{\text{pr}\left(\vec{x}\right) \ \ \overbrace{\text{like}\left(\vec{d}|\vec{x}\right)^{1/T}}^\text{tempered likelihood}}{\text{evi}\left(\vec{d}\right)} , \quad \text{where} \ 1 \leq T < \infty ,
$$ (tempered_posterior)

with the following new component:

| Object | Type | Description |
|-------:|:----:|:------------|
| $T$ | scalar | The "temperature" value that scales the behavior of the likelihood PDF.  "Hotter" temperatures suppress the likelihood PDF so that the posterior is more dependent on the prior.  The "coldest" temperature allowed $T=1$ preserves the original likelihood.

As the temperature $T$ increases and gets "hotter," the likelihood flattens.  In the limit that $T\rightarrow\infty$, the likelihood is completely flattened and its effect vanishes, i.e. $\text{likelihood}\rightarrow 1$.  What this means is that for an infinitly "hot" tempered MCMC, the tempered posterior $\text{post}_{\infty}$ is just the prior itself:

```{margin}
Remember, the evidence is the [normalization factor](./mcmc_basics.md#basic-bayesian) of the posterior PDF.  And the prior itself is a normalized PDF, so in this limit the evidence itself just becomes 1.
```
$$
\textbf{Hot temperature limit:} \quad \text{post}_\infty\left(\vec{x}|\vec{d}\right) = \text{pr}\left(\vec{x}\right) .
$$

As the temperature $T$ decreases and gets "colder," the likelihood returns to its unaltered state.  The "coldest" temperature MCMC at $T=1$ is just the original MCMC of the unaltered posterior:

$$
\textbf{Cold temperature limit:} \quad \text{post}_1\left(\vec{x}|\vec{d}\right) = \text{post}\left(\vec{x}|\vec{d}\right) .
$$

```{important}
Because the cold temperature limit $T=1$ is our orignal posterior PDF, it must always be a part of our PTMCMC set-up!  But how many hotter temperatures, and what the max temperature $T_\text{max}$ that we us in our set-up is, will vary depending on our needs.
```

## The Temperature Acceptance Ratio

It can be difficult to successfully build an MCMC that will efficiently and accurately explore the true underlying parameter space of a posterior when its likelihood PDF is sharply peaked, or multi-modal.  Therefore the motivation for PTMCMC is that by turning up the temperature of the MCMC in the way shown in equation {eq}`tempered_posterior`, the peaks in the likelihood become flatter and broader.  This ultimately improves the chances that proposed jumps will have a higher probability of being accepted!  Hence it will make it easier for hot temperature MCMCs to explore the full parameter space.

However, at the end of the day all we truly care about is the cold $T=1$ temperature, because that is our *original* likelihood.  Hot temperature MCMCs, while easier to explore, are no longer the real posterior.  Therefore we need a way for the benefits of the hotter temperature MCMCs to translate into the cold temperature MCMCs.  So we run multiple copies of our MCMC algorithm, each operating at a different temperature.  The cold temperature MCMCs will likely struggle to explore the full parameter space, but the hot temperature MCMCs will be able to explore the parameter space more easily.

As all of the parallel MCMCs iterate, we will periodically select two temperatures and propose that they swap their current set of parameters.  This will enable each parallel MCMC to effectively run independently, but still share information.  And what this means is that every once in a while, a colder temperature MCMC will be able to make a much larger move in its parameter space than it would have otherwise, simply because it is swapping its current parameter values with the parameter values of a hotter temperature!

The new added criteria to make this swap between parameters looks very similar to our original acceptance ratio criteria {eq}`acceptance_ratio`.  We will call it the "temperature acceptance ratio," and in mathematical notation it is expressed as:

$$
\overbrace{\text{A}_T\left(\vec{x}_i | \vec{x}_j \right)}^\text{temp acceptance ratio} = \ \text{min}\left[ \ \left(\frac{\text{like}\left(\vec{d}|\vec{x}_i\right)}{\text{like}\left(\vec{d}|\vec{x}_j\right)}\right)^{\left(\frac{1}{T_j} - \frac{1}{T_i}\right)}  \ , \quad 1 \ \right] ,
$$ (temperature_acceptance_ratio)

with the following new component:
| Object | Type | Description |
|-------:|:----:|:------------|
| $\text{A}_T\left(\vec{x}_i \| \vec{x}_j \right)$ | function | The "**temperature acceptance ratio**" function.  Take and return whichever is the minimum value between the tempered likelihood ratio terms and $1$ in equation {eq}`temperature_acceptance_ratio`.  It is calculating the probability that the PTMCMC sampling algorithm will "accept" or "reject" the proposed swap of samples $\vec{x}$ between the $i$th temperature MCMC and the $j$th temperature MCMC. |

## Translation to Code: The Heart of the Parallel Tempering Algorithm

Let's translate equation {eq}`temperature_acceptance_ratio` into the actual thing we need to code up.  Similar to the [acceptance ratio translation](./mcmc_basics.md#translation-to-code-the-heart-of-the-mcmc-algorithm), the translation here to code is just a slight adjustment:

```{admonition} Accept swap between parallel tempered samples $\vec{x}_i$ and $\vec{x}_j$ only if:
$$
\left(\frac{\text{like}\left(\vec{d}|\vec{x}_i\right)}{\text{like}\left(\vec{d}|\vec{x}_j\right)}\right)^{\left(\frac{1}{T_j} - \frac{1}{T_i}\right)} \ > \ \mathcal{U}\left[0,1\right] ,
$$(temperature_acceptance_ratio_code)

or, if we take the natural log of both sides of the equation:

$$
    \left(\frac{1}{T_j} - \frac{1}{T_i}\right) \Bigg(\ln\bigg[\text{like}\left(\vec{d}|\vec{x}_i\right)\bigg] - \ln\bigg[\text{like}\left(\vec{d}|\vec{x}_j\right)\bigg]\Bigg) \ > \ \ln\Big[\mathcal{U}\left[0,1\right]\Big] ,
$$(ln_temperature_acceptance_ratio_code)
```

## Pseudo-Code for the PTMCMC Algorithm

We can actually just keep all of the same steps we wrote down before for our [MCMC pseudo-code](./mcmc_basics.md#pseudo-code-for-the-mcmc-algorithm), but add a second set of parallel tempering steps:

```{admonition} The MCMC + Parallel Tempering Algorithm (a high level view)

**MCMC**
1. Initialize a starting parameter sample $\vec{x}_0$ that we will then use to start proposing new samples.
2. Draw a proposed parameter sample $\vec{x}_{i+1}$ based on the current sample $\vec{x}_{i}$.  This will be done through our jump proposal $J\left(\vec{x}_{i+1} | \vec{x}_{i} \right)$.
3. Now use the proposed parameter sample $\vec{x}_{i+1}$ and the current parameter sample $\vec{x}_{i}$ to calculate all of the ratios and terms found in the left-hand side of the acceptance ratio equation {eq}`acceptance_ratio_code`.
4. Draw a random number from the Uniform distribution to give the right-hand side of the acceptance ratio equation {eq}`acceptance_ratio_code`.
5. Compare the two final numbers from steps 3. and 4. to decide whether or not to keep or reject the new proposed sample value.

**Parallel Tempering**
1. At each iteration, loop all of the MCMC steps, but with with tempered likelihoods.
2. Draw a random number from the Uniform distribution to give the right-hand side of the temperature acceptance ratio equation {eq}`temperature_acceptance_ratio_code`.
3. Select two temperatures and their current parameter samples, to calculate the ratio in the left-hand side of the temperature acceptance ratio equation {eq}`temperature_acceptance_ratio`.
4. Compare the two final numbers from steps 2. and 3. to decide whether or not the temperatures swap parameter values.
```

[Parallel Tempered MCMC (PTMCMC)](./schematics/schematics.md#parallel-tempered-mcmc-ptmcmc) shows a **visual schematic** of this general structure.  As always, we just repeat this process for a huge number of iterations until we believe that we have samples that accurately represent our posterior distribution!

## Additional Notes about PTMCMC

{cite:t}`Vousden_2016` is a helpful resource for learning more about PTMCMC.  It is a paper devoted to discussing how to construct more advanced parallel tempering set-ups.

The parallel tempering algorithm steps themselves have some general flexibility in how they are practically implemented:

- The temperature swaps do not have to happen at every step (like what is [visually shown here](./schematics/schematics.md#parallel-tempered-mcmc-ptmcmc)).  Depending on the problem and how many temperatures are being used, it might be beneficial to only periodically propose temperature swaps, to allow the MCMC sampler enough time to explore the local parameter space before it potentially makes large tempered jumps.
- There are different approaches to choosing which temperatures are selected for a proposed swap.  It is typically recommended that the temperatures are adjacent, otherwise the probability of accepting a temperature swap between two widely separated tempered MCMCs might be very low.  But we could cycle from low to high temperatures (as [shown here](./schematics/schematics.md#parallel-tempered-mcmc-ptmcmc)), from high to low, or even randomly.
- The temperatures could be static or dynamic!  Static temperatures are much easier to implement, while dynamically changing temperatures which evolve as the MCMC algorithm progresses are more sophisticated.  Dynamically adjusting temperatures are the main focus of {cite:t}`Vousden_2016`.
