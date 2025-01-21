from trueskillthroughtime import *
import numpy as np
import time
from scipy.stats import norm

N=10000
start = time.time()
means = np.arange(0, 3.5, 0.25)
samples = np.random.normal(means[:, None], 1, (14, N))
higher_indices = np.argmax(samples, axis=0)
counts = np.bincount(higher_indices, minlength=14)/N
print(f"Counts for each Gaussian distribution:\n{counts}")
end = time.time()
print(end-start)

(1-norm(*(Gaussian(3.25,1)-Gaussian(3.0,1))).cdf(0))*\
(1-norm(*(Gaussian(3.25,1)-Gaussian(2.75,1))).cdf(0))*\
(1-norm(*(Gaussian(3.25,1)-Gaussian(2.5,1))).cdf(0))<0.33

