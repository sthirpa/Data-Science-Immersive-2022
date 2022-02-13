- **Required Preparation** will include installing new packages, downloading data, and so on. This is required.
- **Optional Prework** will contain videos, reading, and other resources that may help prepare you for the following week's content. This is optional, but it is recommended that you review the materials shared here!

## Required Preparation:

We will install `pymc3`. Since this library has many dependencies, we will create a **new environment** to minimize conflicts. Follow [these steps:](https://github.com/pymc-devs/pymc3#Installation)


#### Windows
- Run `conda create -n bayes python=3.8 mkl-service libpython m2w64-toolchain scipy matplotlib pandas nb_conda_kernels`
- Run `pip install pymc3` OR `conda install -c conda-forge pymc3`


#### MacOS
- Run `conda create -n bayes python=3.8 scipy matplotlib pandas nb_conda_kernels`	
- Run `conda install -c conda-forge mkl pymc3`
	
#### Linux
- Run `conda create -n bayes python=3.8 scipy matplotlib pandas nb_conda_kernels`
- Run `conda install -c conda-forge pymc3`

To check your install, open a jupyter notebook (switch to your `bayes` kernel) and run the following cell:


```python
import pymc3 as pm

with pm.Model() as model:
    X = pm.Normal("X", mu=3, sd=1)
    Y = pm.Normal("Y", mu=5, sd=2)
    Z = X + Y
    tr = pm.sample(1000, tune=100)
```

- This code can take up to 30 seconds to run, and will run with warnings, but no errors.
- IF YOU GET AN ERROR, PLEASE CONSULT AN INSTRUCTOR, BUT ONLY AFTER TAKING THE TIME TO ATTEMPT TO SOLVE THE PROBLEM YOURSELF! The dependencies for these packages can be a nightmare, and if you can error, it's likely yours will be different from your neighbor's. Google will be your friend here.

## Optional Prework:

- Check out the Naive Bayes lab, as we'll explore using Bayes' Theorem this week.
- Check out our continuous and discrete distributions lectures from earlier weeks, as we’ll be using lots of statistical distributions!
- Sections 0.2 and 1.1-1.5 in http://greenteapress.com/wp/think-bayes/
- As a good resource, check out [**Bayesian Methods for Hackers**](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/). It's a notebook-style online book that efficiently teaches Bayesian statistics through some pretty compelling real-world applications. It does a great job teaching Bayesian statistics with minimal mathematical background. Highly recommended. - Tim Book
