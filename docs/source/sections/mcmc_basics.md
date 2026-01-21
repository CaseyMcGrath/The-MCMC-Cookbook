# MCMC Basics

One aspect of trying to learn and understand MCMC and the general Bayesian analysis framework that I often have found challenging is simply the notation.  Everyone has a *slightly* different notation.  So going between papers and texts, sometimes I find I have to be really careful to translate the same idea communicated in different notations.

But I often feel that the notation can be made clearer and more effective.  Hopefully I can achieve that here (it is a very intentional goal for **The MCMC Cookbook**) - but I'm always open to constructive criticism!

What follows will not be an exhaustive review of Bayesian analysis - hopefully you have already learned that or are in the process of learning it!  But we will try to establish a notation that is easier to parse, and then use that to dive into our look at how the MCMC algorithm works.

## Basic Bayesian

From **Bayes' Theorem** the main formula that the entire MCMC algorithm stems from is:
```{margin}
Remember to read the "bar" notation $\left(a | b\right)$ as "$a$ given $b$."
```

$$
\overbrace{\text{post}\left(\vec{x}|\vec{d}\right)}^\text{posterior} = \frac{\overbrace{\text{pr}\left(\vec{x}\right)}^\text{prior} \ \ \overbrace{\text{like}\left(\vec{d}|\vec{x}\right)}^\text{likelihood}}{\underbrace{\text{evi}\left(\vec{d}\right)}_\text{evidence}} ,
$$ (bayes_theorem)

with each of the following components:

```{margin} Probability Density Functions (PDFs)
Remember that to be a PDF, it must be normalized. It is a *probability density*, the integral over all possible values of the input must add up to $1$.  We have three PDFs in this table, but in the general literature only one of them has a uniquely *named* normalization factor - the posterior's evidence.  However, both the prior and the likelihood are also normalized functions by themselves.  Sometimes in MCMC coding and analysis, we can get away with removing their normalization factors because they aren't explicitly needed.  In fact, we often don't need the evidence when coding up a MCMC sampler either!  We will do our best to be careful and explicitly acknowledge when we do and don't need their normalizations, and explain why.
```
| Object | Type | Description |
|-------:|:----:|:------------|
| $\vec{x}$ | vector | The "list" (or vector or array, how ever you want to think about it!) of the parameters in our model.  This is the thing we want our MCMC to estimate!  If our model only has a single parameter, then this is just a scalar. |
| $\vec{d}$ | vector | The "list" of the data we have observed.  This is the input to our model to inform our MCMC. |
| $\text{post}\left(\vec{x}\|\vec{d}\right)$ | PDF | The "**posterior**" PDF.  It is a function of the model parameters, given the data observed.  At the end of the day, our entire MCMC algorithm is being built to figure this thing out!  We don't know it ahead of time. |
| $\text{pr}\left(\vec{x}\right)$ | PDF | The "**prior**" PDF.  It is a function of the model parameters, and represents our intial belief of how the parameters should be distributed.  This is usually the easiest thing to write down when building our MCMC. |
| $\text{like}\left(\vec{d}\|\vec{x}\right)$ | PDF | The "**likelihood**" PDF.  It is a function of the observed data, given a set of specified model parameters.  This is usually the most complicated thing we have to write down when building our MCMC. |
| $\text{evi}\left(\vec{d}\right)$ | normalization | The "**evidence**" is the normalization factor of the posterior PDF.  It is the integral of the prior and the likelihood over the entire possible parameter space.  That is: $ \text{evi}\left(\vec{d}\right) = \int \ \text{pr}\left(\vec{x}\right) \ \text{like}\left(\vec{d}\|\vec{x}\right) \ \text{d}\vec{x} $ |

## The Acceptance Ratio

The fundamental idea behind an MCMC sampler is that it is an algorithm that draws samples from the posterior distribution - the thing we want to know.  The way it works is that at every step in the sampling process, it must propose a new position in our parameter space, i.e. a new sample or a new draw.  The sampler then has a criteria it must meet in order to keep that sample - otherwise the sampler simply rejects the sample altogether and continues on to a new proposed sample.

The criteria is the "acceptance ratio."  In mathematical notation, you will often see it in the literature written something like the following:
```{margin} Critical observation!
Notice how in the first two terms in the acceptance ratio, the respective prior and likelihood ratios, they are ratios of the proposed parameter sample (in the numerator) to the current parameter sample (in the denominator).  However, in the third term, the jump proposal ratio, *it is the reverse!*  It is the ratio of the current parameter sample to the proposed parameter sample.  Try to digest this, because I think it is easy to lose this understanding in the notation, which is explained more below.
```
$$
\overbrace{\text{A}\left(\vec{x}_{i+1} | \vec{x}_i \right)}^\text{acceptance ratio} = \text{min}\left[ \ \frac{\text{pr}\left(\vec{x}_{i+1}\right)}{\text{pr}\left(\vec{x}_{i}\right)} \ \frac{\text{like}\left(\vec{d}|\vec{x}_{i+1}\right)}{\text{like}\left(\vec{d}|\vec{x}_{i}\right)} \ \frac{\overbrace{\text{J}\left(\vec{x}_{i}|\vec{x}_{i+1}\right)}^\text{jump proposal}}{\text{J}\left(\vec{x}_{i+1}|\vec{x}_{i}\right)} \quad , \quad 1 \ \right] ,
$$ (acceptance_ratio)

with each of the following new components:
| Object | Type | Description |
|-------:|:----:|:------------|
| $\vec{x}_{i}$ | vector | The "current" parameter sample.  An MCMC algorithm is just a giant loop, so we are constantly iterating new samples - hence we need a notation to denote "current" and "proposed."  We will use the indices "$i$" to track this. |
| $\vec{x}_{i+1}$ | vector | The "proposed" parameter sample.  This is the new sample we are proposing to add to our MCMC's drawn samples. |
| $\text{A}\left(\vec{x}_{i+1} \| \vec{x}_i \right)$ | function | The "**acceptance ratio**" function.  Take and return whichever is the minimum value between those ratio terms and $1$ in equation {eq}`acceptance_ratio`.  It is calculating the probability that the MCMC sampling algorithm will "accept" or "reject" the proposed sample $\vec{x}_{i+1}$ given the current sample $\vec{x}_{i}$. |
| $\text{J}\left(\vec{a}\|\vec{b}\right)$ | PDF | The "**jump proposal**" PDF.  It is a function of the input parameter $\vec{a}$ given some starting parameter $\vec{b}$.  I would argue that the true "art" of becoming a master at MCMC is understanding how to create jump proposals that will best suite your problem and enable your MCMC algorithm to most efficiently search the *full* parameter space. More on that to come! |


```{important}
The **jump proposal** will actually serve two very important aspects of our MCMC algorithm:
1. It needs to be able to *generate* new random parameter sample draws to then feed into the algorithm.
2. It needs to be able to calculate the PDF value of a proposed sample given the current sample.

Keep this in the back of your mind going forward!
```

## Translation to Code: The Heart of the MCMC Algorithm

Ok seeing the acceptance ratio equation {eq}`acceptance_ratio` is great, but as someone who is now interested in turning that into actual *code*, I initially found that notation to not be the most helpful.  Ultimately the translation of equation {eq}`acceptance_ratio` into actual code - the way that it will look when you start writing your MCMC algorithm - will be the following.  It's just a slight adjustment, but in my mind conceptually easier to understand how to implement:

```{admonition} Accept new proposed parameter sample $\vec{x}_{i+1}$ only if:
$$
\frac{\text{pr}\left(\vec{x}_{i+1}\right)}{\text{pr}\left(\vec{x}_{i}\right)} \ \frac{\text{like}\left(\vec{d}|\vec{x}_{i+1}\right)}{\text{like}\left(\vec{d}|\vec{x}_{i}\right)} \ \frac{\text{J}\left(\vec{x}_{i}|\vec{x}_{i+1}\right)}{\text{J}\left(\vec{x}_{i+1}|\vec{x}_{i}\right)} \ > \ \mathcal{U}\left[0,1\right] ,
$$(acceptance_ratio_code)

or, if we take the natural log of both sides of the equation:

$$
\begin{align}
    \ln\Bigg(\text{pr}\left(\vec{x}_{i+1}\right) - \text{pr}\left(\vec{x}_{i}\right)\Bigg) \ + \ \ln\Bigg(\text{like}\left(\vec{d}|\vec{x}_{i+1}\right) - \text{like}\left(\vec{d}|\vec{x}_{i}\right)\Bigg) \\  
    + \ \ln\Bigg(\text{J}\left(\vec{x}_{i}|\vec{x}_{i+1}\right) - \text{J}\left(\vec{x}_{i+1}|\vec{x}_{i}\right)\Bigg) \ > \ \ln\Big(\mathcal{U}\left[0,1\right]\Big) ,
\end{align}
$$(ln_acceptance_ratio_code)
```

with each of the following new components:
| Object | Type | Description |
|-------:|:----:|:------------|
| $\mathcal{U}\left[0,1\right]$ | PDF sample | A random draw from the Uniform distribution (defined on the interval from $[0,1]$). |

## Pseudo-Code for the MCMC Algorithm

Now we have all the necessary pieces, so let us just write down a very simple set of steps that our MCMC algorithm will need to go through.

```{margin}
Steps 1, 4, and 5 are pretty trivial.  Most of what makes setting up a new MCMC challenging I think is in steps 2 and 3 here.  But hopefully once we start to break all of this down, understanding steps 2 and 3 will become easier!
```
```{admonition} The MCMC Algorithm (a high level view)
1. Initialize a starting parameter sample $\vec{x}_0$ that we will then use to start proposing new samples.
2. Draw a proposed parameter sample $\vec{x}_{i+1}$ based on the current sample $\vec{x}_{i}$.  This will be done through our jump proposal $J\left(\vec{x}_{i+1} | \vec{x}_{i} \right)$.
3. Now use the proposed parameter sample $\vec{x}_{i+1}$ and the current parameter sample $\vec{x}_{i}$ to calculate all of the ratios and terms found in the left-hand side of the acceptance ratio equation {eq}`acceptance_ratio_code`.
4. Draw a random number from the Uniform distribution to give the right-hand side of the acceptance ratio equation {eq}`acceptance_ratio_code`.
5. Compare the two final numbers from steps 3. and 4. to decide whether or not to keep or reject the new proposed sample value.
```

Now we just repeat this process for a huge number of iterations, until we believe that we have samples that accurately represent our posterior distribution!

