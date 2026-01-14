# Toy Problems

Personally, I'm a fan of starting easy.  Here we will illustrate some (hopefully!) *simple toy problem examples* that we will use throughout the course of **The MCMC Cookbook**.

## "The Gaussian Bump"

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

**Dataset**

We collect a total of $N_t$ observed data and store it in an array $\vec{d}$.  And let's say that the observation uncertainty is the same for every data point.  In other words, the underlying noise in our measurements is just "white noise" with standard deviation $\sigma_n$ - none of our observations are correlated with each other.

**Likelihood**

With the description of the dataset's noise in mind, we might therefore expect that if we could perfectly extract just the noise from our data, and plot a histogram of it, the noise should just be randomly distributed following a Gaussian distribution.  Since the data $\vec{d}$ is just the noise $\vec{n}$ plus our underlying signal/model $\vec{M}$, the noise is therefore $\vec{n} = \vec{d} - \vec{M}$, and we could write our likelihood function as a multivariate Gaussian:

$$
\begin{align}
    \text{like}\left(\vec{d} | \vec{x} \right) &= \mathcal{N} \exp\left[-\frac{1}{2}\frac{\left(\vec{d} - \vec{M}(t,\vec{x}) \right)^2}{\sigma_n^2}\right] \\
    &= \mathcal{N} \exp\left[-\frac{1}{2}\left(\frac{\left(d_1 - M(t_1,\vec{x}) \right)^2}{\sigma_n^2} + \frac{\left(d_2 - M(t_2,\vec{x}) \right)^2}{\sigma_n^2} + \cdots \right)\right] \\
    &= \mathcal{N} \exp\left[-\frac{1}{2}\sum_{i}^{N_t}\frac{\left(\vec{d}_i - \vec{M}(t_i,\vec{x}) \right)^2}{\sigma_n^2}\right] \\
    &= \mathcal{N} \prod_{i}^{N_t} \exp\left[-\frac{1}{2}\frac{\left(\vec{d}_i - \vec{M}(t_i,\vec{x}) \right)^2}{\sigma_n^2}\right] \\
\end{align}
$$

Note we've expressed the same equation in several different formats here, just so that you hopefully understand that all of these notations mean the same thing!  The normalization factor $\mathcal{N}$ of this PDF is:

$$
\mathcal{N} = \frac{1}{\sqrt{\left(2\pi \sigma_n^2\right)^{N_t}}}
$$





