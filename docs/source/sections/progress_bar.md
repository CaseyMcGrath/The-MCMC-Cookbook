# Convenience: Progress Tracking Bar

Ok this is probably more of a public service announcement than anything else!  For a long time I did not know about the *TQDM* python package, but I wish I had!  You can easily plug it into your code where you might want some kind of progress tracker - something that I just like to have for my MCMCs as a convencience!  Basically you just initiate it whenever you want to track the progress of a `for` loop.

**TQDM Example**


```python
from tqdm import tqdm  # progress bar
import numpy as np
```

Let's just generate a bunch of random numbers and see how long it takes!


```python
N = 10_000_000

for i in tqdm(range(N)):
    np.random.rand()
```

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 10000000/10000000 [00:01<00:00, 5125529.09it/s]


Look at that, a nice little progress bar that shows you:
- how long the loop was,
- how long it took to complete the loop (and while the loop is running, it estimates how much time is left until it is finished!),
- and the rate of loops/second (another useful tool for looking for ways for speeding up your code!).
