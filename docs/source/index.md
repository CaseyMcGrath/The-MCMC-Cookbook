

Recipes for Crafting MCMCs
==========================

When I first began my journey into the world of Markov Chain Monte Carlo (MCMC) samplers (in graduate school), I used a pre-built sampler.  I downloaded it from github and learned how to use it through a combination of practice on toy problems and reading through the documentation.  And then I tried to use it to solve the very specific problem for my research.  I suspect that many people begin their own MCMC journeys in a similar way.

But there were problems.  The first problem was that I never really understood *how* an MCMC worked.  I understood it on a conceptual level, but the algorithm I had downloaded was a black box.  Modifying the code was not really an option for me, unless I wanted to spend all of my time learning every last detail of how someone else's MCMC code worked.  And in my mind it was easy to argue that this would take an enormous amount of time, for potentially little to no pay-off.

The second problem was that I often quickly ran into all sorts of challenges using the sampler.  My problem would be far too complex to easily solve with the default settings of the black box MCMC.  So I would spend endless hours changing little nobs and dials in the sampler, re-running it, trying to understand if my results were getting better.  And I was almost never satisfied with the results.

I felt I had reached my limit.  The problems I wanted to solve were too complicated, and while I did know a handful of individuals who I did coonsider "masters of the art of MCMC," they were often to busy to truly apprentice me to their level.  And I was never fully satisfied with the reading materials I could find for trying to teach myself.  Academic papers are often too technical, the jargon is too dense, and I would find they were not written at a level that I could understand.

The truth is that MCMCs are simple alorithms, with enormous amounts of variation.  The MCMC sampler that will work best on the problem you want to solve will likely depend on the problem you want to solve, and it's specific nuances and complexities.  *This* is why (I think) MCMCs are hard to master.  If your problem is relatively simple, then using someone else's black box MCMC might work just fine for you!  This is certainly no slight against the wonderful developers of so many helpful MCMC tools out there, because they are excellent tools, and when they work well, you should use them!

But after I spent over a decade of banging my head against my computer and reaching my threshold of understanding, I finally was given the hard advice from a good friend (someone who I did consider a master at MCMC) that I needed to hear:

> "To really learn the art of MCMC, you need to build one from scratch."

So that is the purpose of this cookbook: to develop from scratch MCMC samplers for various problems, to learn about different ingredients that often go into different MCMC samplers, and to learn which ingredients to add into your MCMC depending on the problem you want to solve.  That way, hopefully, you can learn to build your own recipe and cook exactly the MCMC that you need!





```{toctree}
:maxdepth: 2
:caption: Contents:

sections/
```