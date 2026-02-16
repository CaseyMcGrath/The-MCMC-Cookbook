# Toy Problems

Personally, I'm a fan of starting easy.  Here we will illustrate some (hopefully!) *simple toy problem examples* that we will use throughout the course of **The MCMC Cookbook**.

## "The Bump"

**Description**

Let's imagine we are making an observation of *something* over a period of time, for a total of $N_t$ data points.  And we observe a bell-shaped rise and fall in that quantity's measurement (that resembles a "bump").  So we decide to fit a Gaussian function to the data.

To keep this example simple, we are going to analyze the data in the time-domain.  So we'll need to write a time-domain likelihood function that uses this model and inputs the observed data.  We'll also need to choose priors for these model parameters.

**Model** 

$$
M(t) = A \exp\left[-\frac{1}{2}\frac{(t-t_0)^2}{\sigma^2}\right]
$$ (bump_model)

**Parameters**
- $A$ = amplitude
- $t_0$ = central time
- $\sigma$ = bump width (the standard deviation of the Gaussian)

We will assume all three parameters are **independent**.

**Dataset**

We collect a total of $N_t$ observed data and store it in an array $\vec{d}$.  And let's say that the observation uncertainty is the same for every data point.  In other words, the underlying noise in our measurements is just "white noise" with standard deviation $\sigma_n$ - none of our observations are correlated with each other.

Here is visual example of what this signal and dataset might look like:

![png](bump_dataset_example.png)

**Likelihood**

With the description of the dataset's noise in mind, we might therefore expect that if we could perfectly extract just the noise from our data, and plot a histogram of it, the noise should just be randomly distributed following a Gaussian distribution.  Since the data $\vec{d}$ is just the noise $\vec{n}$ plus our underlying signal/model $\vec{M}$, the noise is therefore $\vec{n} = \vec{d} - \vec{M}$, and we could write our likelihood function as a multivariate Gaussian:

$$
\begin{align}
    \text{like}\left(\vec{d} | \vec{x} \right) &= \mathcal{N} \exp\left[-\frac{1}{2}\frac{\left(\vec{d} - \vec{M}(t,\vec{x}) \right)^2}{\sigma_n^2}\right] \\
    &= \mathcal{N} \exp\left[-\frac{1}{2}\left(\frac{\left(d_1 - M(t_1,\vec{x}) \right)^2}{\sigma_n^2} + \frac{\left(d_2 - M(t_2,\vec{x}) \right)^2}{\sigma_n^2} + \cdots \right)\right] \\
    &= \mathcal{N} \exp\left[-\frac{1}{2}\sum_{i}^{N_t}\frac{\left(d_i - M(t_i,\vec{x}) \right)^2}{\sigma_n^2}\right] \\
    &= \mathcal{N} \prod_{i}^{N_t} \exp\left[-\frac{1}{2}\frac{\left(d_i - M(t_i,\vec{x}) \right)^2}{\sigma_n^2}\right]
\end{align}
$$(time_domain_likelihood)

Note we've expressed the same equation in several different formats here, just so that you hopefully understand that all of these notations mean the same thing!  The normalization factor $\mathcal{N}$ of this PDF is:

$$
\mathcal{N} = \frac{1}{\sqrt{\left(2\pi \sigma_n^2\right)^{N_t}}}
$$

**Prior**

For the central time $t_0$, a reasonable prior is a **uniform prior**.  We have some dataset with a specific start and end time, and any time between those two values is equally reasonable as a potential value for $t_0$.

For both the amplitude $A$ and bump width $\sigma$ parameters, we will give each a **log-uniform prior**.  Unlike $t_0$, let's say we don't really have a reason to think there would exist hard "boundaries" on $A$ or $\sigma$ (like with the start and end times of our dataset).  Moreover, we might expect there could exist a dynamic range of bump amplitudes and widths.  We will say it is reasonable that small bumps (both in amplitude and/or width) are expected to occur more frequently, while large bumps are less common.  So with this in mind, choosing a log-uniform prior puts more probability on small duration bumps than it does large duration bumps, and can span a dynamic range.

And since we are working with independent parameters, the joint prior will just be the product of the individual parameter priors


$$
\text{pr}\left(\vec{x}\right) = \text{pr}(A) \ \text{pr}(t_0) \ \text{pr}(\sigma)
$$

## "The Wave"

**Description**

Let's imagine we are making an observation of *something* over a period of time, for a total of $N_t$ data points.  And we observe a periodic change in that quantity's measurement (that resembles a sinusoid or "wave").  So we decide to fit a cosine function to the data.

To keep this example simple, we are going to analyze the data in the time-domain.  So we'll need to write a time-domain likelihood function that uses this model and inputs the observed data.  We'll also need to choose priors for these model parameters.

**Model** 

$$
M(t) = A \cos\left(\phi_0 + 2\pi f_0 t\right)
$$ (wave_model)

**Parameters**
- $A$ = amplitude
- $\phi_0$ = initial phase
- $f_0$ = frequency

We will assume all three parameters are **independent**.

**Dataset**

We collect a total of $N_t$ observed data and store it in an array $\vec{d}$.  And let's say that the observation uncertainty is the same for every data point.  In other words, the underlying noise in our measurements is just "white noise" with standard deviation $\sigma_n$ - none of our observations are correlated with each other.

Here is visual example of what this signal and dataset might look like:

![png](wave_dataset_example.png)

**Likelihood**

We are using the same description of the observations/dataset as we did for [The Bump](#the-bump), so let's use the same exact likelihood equation {eq}`time_domain_likelihood` as we did for that model (just replacing our model $M(t)$ with the new wave model).

**Prior**


```{margin}
Meaning, for example, if I fit to the model $\phi_0 = 2$, the values $\phi_0 = 2 + 2\pi$, $\phi_0 = 2 + 4\pi$, $\phi_0 = 2 - 12\pi$, etc. will all produce identical results.
```
For the phase parameter $\phi_0$, we know that with sinusoidal functions, their phase wraps around the interval $\left[0, 2\pi\right)$.  So the initial phase parameter is a **cyclic parameter**, and also exists within this cyclic boundary.  Therefore a reasonable prior is a **uniform prior** on this interval (any time between these two values is equally reasonable as a potential value for $\phi_0$).

For both the amplitude $A$ and frequency $f_0$ parameters, we will give each a **log-uniform prior**.  Unlike $\phi_0$, let's say we don't really have a reason to think there would exist hard "boundaries" on $A$ or $f_0$ (like with the $\left[0, 2\pi\right)$ cyclic interval of $\phi_0$).  Moreover, we might expect there could exist a dynamic range of wave amplitudes and frequencies.  We will say it is reasonable that small, low frequency waves are expected to occur more frequently than large, high frequency waves.

And since we are working with independent parameters, the joint prior will just be the product of the individual parameter priors

$$
\text{pr}\left(\vec{x}\right) = \text{pr}(A) \ \text{pr}(\phi_0) \ \text{pr}(f_0)
$$
