# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook, Float
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
import pandas as pd
from docs.examples.utils import clean_legend

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
from gpjax.typing import Array
from sklearn.preprocessing import StandardScaler

key = jr.PRNGKey(42)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


#use a regular grid of scalar inputs
NPoints = 400
xQuery = jnp.linspace(0,1,NPoints)

kernel = gpx.kernels.RBF(lengthscale = jnp.array(1), variance = jnp.array(1))


xTraining = jnp.array([2,3.5,5,8])
fTraining = 3*jnp.array([0.3,0.5,0.4,1.2])








