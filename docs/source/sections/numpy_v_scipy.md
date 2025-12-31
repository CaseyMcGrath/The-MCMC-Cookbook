# Random Numbers: *NumPy* vs. *SciPy*

To build an MCMC, we need to be able to draw random numbers... a lot.

More specifically, we want to be able to choose specific probability distributions, and generate both random values (RVs) from those distributions, as well as the value of the probability density function (PDF) at those RVs.

Two easy-to-use packages for this are [*NumPy*](https://numpy.org/doc/stable/reference/random/legacy.html) and [*SciPy*](https://docs.scipy.org/doc/scipy/reference/stats.html).  However, there are pros and cons to consider when making your choice.  Both packages let you draw random samples from defined PDFs very easily.  *NumPy* does it faster (see below).  However, *SciPy* also let's you compute the PDF values incredibly easily, and has many, many more useful features built into it as compared to *NumPy*.

So for much greater easy of implementation benefits, we are going to stick to using *SciPy* in **The MCMC Cookbook**.  When you go to build your own MCMC, you could potentially speed up some of our computations if you switch back to NumPy for drawing your RVs.  You'll have to decide if it is worth it for you and your own application!


## Summary:
- *NumPy* random draws are faster than *SciPy*
- But *SciPy* has many useful features built into it because of it's class structure (e.g. you can quickly draw a random variable using `.rvs()`, and compute the value of a PDF using `.pdf()` for many well known distributions!)

## Checking the Speed of *NumPy* over *SciPy*

Just a simple check - let's generate a bunch of random draws from a multivariate normal distribution using both *NumPy* and *SciPy* and see how fast they are.


```python
import time
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
start_time = time.time()      # Record the start time

for i in tqdm(range(N)):
    np.random.multivariate_normal(mean, cov)

end_time = time.time()        # Record the end time

elapsed_time = end_time - start_time
print(f"The loop took {elapsed_time:.4f} seconds.")
```

    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000000/1000000 [00:25<00:00, 39689.82it/s]

    The loop took 25.2077 seconds.


    


**SciPy Test**


```python
start_time = time.time()      # Record the start time

for i in tqdm(range(N)):
    scipy.stats.multivariate_normal(mean, cov).rvs()

end_time = time.time()        # Record the end time

elapsed_time = end_time - start_time
print(f"The loop took {elapsed_time:.4f} seconds.")
```

    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000000/1000000 [01:14<00:00, 13452.91it/s]

    The loop took 74.3342 seconds.


    


Since MCMCs have to draw lots and lots of RVs, you can potentially gain a pretty significant boost in computation speed by switching your RV draws to using *NumPy*.  Just something to keep in mind!
