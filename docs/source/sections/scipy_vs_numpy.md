# Random Numbers: *SciPy* vs. *NumPy*

To build an MCMC, we need to be able to draw random numbers... a lot.

More specifically, we want to be able to choose specific probability distributions, and 
1. generate random values (RVs) from those distributions,
2. and calculate the value of the probability density function (PDF) at those RVs.

```{margin}
*SciPy* uses a "class" structure for its probability densities.  Meaning you must start by instantiating the PDF you want (and *SciPy* has a large library to choose from), then with the PDF object you can call in-built functions such as `.rvs()` to generate a random value, or `.pdf()` to calculate the probability density at a specific value.
```
Two easy-to-use packages for this are [*SciPy*](https://docs.scipy.org/doc/scipy/reference/stats.html) and [*NumPy*](https://numpy.org/doc/stable/reference/random/generator.html).  Both packages let you draw random samples from defined PDFs very easily.  However, *SciPy* has a coding structure with many more built in features which are incredibly useful, namely it can do the two things listed above quickly and easily.  So for much greater ease of implementation benefits, we are mostly going to stick to using *SciPy* throughout **The MCMC Cookbook**.

## Speed Test Comparison

Let's just do a simple check to see if either *NumPy* or *SciPy* have any advantages over one another in terms of speed.  Because MCMC algorithms require such a high volume of random number generations, any advantage in speed will be an important consideration!

For this test, let's generate a bunch of random draws from a multivariate normal distribution using both packages and see how fast they are.


```python
# import time
from tqdm import tqdm  # progress bar
import numpy as np
import scipy.stats
```


```python
# Multivariate Normal distribution - mean and covariance matrix
mean = np.array([0,0])
cov  = np.array([[1,0],[0,1]])

# number of random draws
N = 1_000_000
```

**NumPy Test**


```python
# Initialize the "random number generator"
rng = np.random.default_rng()

for i in tqdm(range(N), bar_format='{l_bar}{bar:30}{r_bar}'):
    rng.multivariate_normal(mean, cov)
```

    100%|██████████████████████████████| 1000000/1000000 [00:25<00:00, 39956.79it/s]


**SciPy Test**


```python
# Instantiate the PDF
multivariate_normal = scipy.stats.multivariate_normal(mean, cov)

for i in tqdm(range(N), bar_format='{l_bar}{bar:30}{r_bar}'):
    multivariate_normal.rvs()
```

    100%|██████████████████████████████| 1000000/1000000 [00:26<00:00, 38280.65it/s]


Ok so both packages seem pretty evenly matched, at least just going off of this simple speed test!

````{warning}
So I want to highlight a mistake that I personally made!  And it was an important lesson because it was an easy fix and it sped up my code!

Because the *SciPy* stats package is class-based, when possible, try to instantiate your probability density function *outside* of any loop!  There will always be some amount of computational overhead under the hood for instantiating the class the first time, but once it has been created, calling its functions repeatedly is fast!

The first time I did this, the mistake I was making looked like the following:

```python
for i in tqdm(range(N), bar_format='{l_bar}{bar:30}{r_bar}'):
    scipy.stats.multivariate_normal(mean, cov).rvs()   # <-- I'm instantiating the class *every time I run the loop!*
```
    100%|██████████████████████████████| 1000000/1000000 [01:14<00:00, 13461.81it/s]

Compared to what I did above, this is *significantly* slower!

We won't always be able to do this!  Sometimes we will necessarily have to instantiate the class inside of the loop.  Actually, if speed is very critical for your problem, in those cases using *NumPy* for the random number generation might actually be more beneficial - if it makes sense.  So it is at least something to keep in the back of your mind, and to test as you go!
````
