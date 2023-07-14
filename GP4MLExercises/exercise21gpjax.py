# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
import pandas as pd

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
import numpy as np
np.random.seed(2)
key = jr.PRNGKey(5)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


#define the kernel that describes the prior, encoding information about how we think the gaussian process should look. (Also set the prior mean to 0)
kernel = gpx.kernels.RBF(lengthscale =jnp.array(1.0), variance = jnp.array(1.0))
meanf = gpx.Zero()

#define the training data, which the prior will be conditioned on
xTraining = jnp.array([2.0,3.5,5.0,8.0]).reshape(-1, 1)
fTraining = 3*jnp.array([0.3,0.5,0.4,1.2]).reshape(-1, 1)
D = gpx.Dataset(X=xTraining, y=fTraining)

#choose a prior which is a gaussian process
prior = gpx.Prior(mean_function=meanf, kernel=kernel) 

#choose a gaussian likelihood (given the training data, how likely we were to observe the training points if we assume this likelihood is gaussian)
likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=10.0) #(we only used D.n (number of points {xi,yi} in gpx.Gaussian)
#units of obs_noise are [variance] ?

#compute the posterior, which we now know more about as we introduce training data
posterior = prior * likelihood 

# query the posterior at the test points given training data (our best knowledge so far given the data and assuming gaussian likelihood)
NPoints = 40
xQuery = jnp.linspace(0,10, NPoints).reshape(-1, 1)
latent_dist = posterior.predict(xQuery, train_data=D) # first time we have used the training data is here!! 

#latent = unobserved. we never explicitly observe the conditioned mean
latent_mean = latent_dist.mean().reshape(-1, 1)
latent_std = latent_dist.stddev().reshape(-1, 1)

#sample the posterior (candidate) functions at each point in our NPoints-D gaussian
NSamples = 4
samples = latent_dist.sample(seed=key, sample_shape=(NSamples,))


fig, ax = plt.subplots(1,1, figsize = (5,3))
ax.plot(xQuery, latent_mean, color = 'gray', alpha = 1, label = '$\mu$')
for i in range(NSamples-1):
    ax.plot(xQuery.squeeze(),samples[i], color = 'red', alpha = 0.3)
ax.plot(xQuery.squeeze(),samples[-1], color = 'red', alpha = 0.3, label = 'GP')
ax.fill_between(
    xQuery.squeeze(),
    latent_mean.squeeze() - 2*latent_std.squeeze(),
    latent_mean.squeeze() + 2*latent_std.squeeze(),
    alpha=0.2,
    label="$\mu \pm 2\sigma$",
    color='gray',
)
ax.plot(xTraining, fTraining,'+' , ms = 15, color = 'black' , label = 'training data')
ax.legend(loc = 'best')
plt.show()