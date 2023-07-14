# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)

from dataclasses import dataclass
from typing import Dict

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
import matplotlib.pyplot as plt
import numpy as np
import optax as ox
from simple_pytree import static_field
import tensorflow_probability.substrates.jax as tfp

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base.param import param_field

key = jr.PRNGKey(123)
tfb = tfp.bijectors
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]


@dataclass
class TiedDownWiener(gpx.kernels.AbstractKernel):
    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        K = jnp.minimum(x,y)-x*y
        return K.squeeze() 

# define the kernel that describes the prior, encoding information about how we think 
# the gaussian process should look. (Also set the prior mean to 0)
kernel = TiedDownWiener()
meanf = gpx.Zero()

#define the training data, which the prior will be conditioned on

xTraining = jnp.array([0.0, 0.3,0.35,0.4,0.5,0.6,0.8,1.0]).reshape(-1, 1)
fTraining = jnp.array([0.0,0.35,1.5,0.8,0.0,-0.3,0.4,0.0]).reshape(-1, 1)
D = gpx.Dataset(X=xTraining,y=fTraining) #put the training data in a Dataset object

#construct the prior, encoding our knowledge of the data thus far
WeinerPrior = gpx.Prior(mean_function=meanf, kernel=kernel)

#construct the likelihood, which we assume to be gaussian (no observation noise for now)
WeinerLikelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=0.0)

#given our data, we condition the posterior to depend on the prioir and likelhood using 
#bayes thm
WeinerPosterior = WeinerPrior * WeinerLikelihood #curious if the order matters?

#Query the posterior at our points of interest given the training data to obtain the latent distribution
NPoints = 400
xQuery = jnp.linspace(0.0,1.0,NPoints).reshape(-1,1) #don't understand this reshaping

#get the latent (unobserved,mean) distribution
#conditioned mean at the query points (?)
LatentDist = WeinerPosterior.predict(xQuery, train_data=D) 
LatentMean =LatentDist.mean().reshape(-1,1)
LatentStd = LatentDist.stddev().reshape(-1,1)
LatentDist.stddev().reshape(-1,1)

#sample the posterior distribution (Npoints-D Gaussian) for a candidate function
NSamples = 3
samples = LatentDist.sample(seed=key, sample_shape=(NSamples,)).squeeze()

#plot
fig, ax = plt.subplots(1,1, figsize = (5,3))
ax.plot(xQuery, LatentMean.squeeze(), color = 'gray', alpha = 1, label = '$\mu$')
for i in range(NSamples-1):
    ax.plot(xQuery.squeeze(),samples[i], color = 'red', alpha = 0.3)
ax.plot(xQuery.squeeze(),samples[-1], color = 'red', alpha = 0.3, label = 'GP')
ax.fill_between(
    xQuery.squeeze(),
    LatentMean.squeeze() - 2*LatentStd.squeeze(),
    LatentMean.squeeze() + 2*LatentStd.squeeze(),
    alpha=0.2,
    label="$\mu \pm 2\sigma$",
    color='gray',
)
ax.plot(xTraining, fTraining,'+' , ms = 15, color = 'black' , label = 'training data')
ax.legend(loc = 'best')
plt.show()