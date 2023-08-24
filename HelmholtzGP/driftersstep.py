from jax.config import config

config.update("jax_enable_x64", True)
from dataclasses import dataclass

from jax.config import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
from matplotlib import rcParams
import matplotlib.pyplot as plt
import optax as ox
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
colors = rcParams["axes.prop_cycle"].by_key()["color"]

#import
import pickle
#reading data
pickle_in = open('driftersmodel.pkl', 'rb')
data = pickle.load(pickle_in)

# opt_helmholtz_posterior, helmholtz_mean, 
# helmholtz_std, pos_train3d
opt_posterior = data[0]
mean = data[1]
std = data[2]
pos = data[3]
DT = data[4]

def latent_distribution(opt_posterior, pos_3d):
    latent = opt_posterior.predict(pos_3d, train_data=DT)
    latent_mean = latent.mean()
    latent_std = latent.stddev()
    return latent_mean, latent_std

mean, std = latent_distribution(opt_posterior, pos)


#import
import matplotlib.pyplot as plt
#make figure
fig, ax = plt.subplots(1,1, figsize =(10,6))
#plt.savefig('.png', dpi = 600)
plt.show()