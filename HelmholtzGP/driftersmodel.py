# %% [markdown]
# # Gaussian Processes for Vector Fields and Ocean Current Modelling
#
# In this notebook, we use Gaussian processes to learn vector valued functions. We will be
# recreating the results by [Berlinghieri et. al, (2023)](https://arxiv.org/pdf/2302.10364.pdf) by an
# application to real world ocean surface velocity data, collected via surface drifters.
#
# Surface drifters are measurement devices that measure the dynamics and circulation patterns of the world's oceans. Studying and predicting ocean currents are important to climate research, for example forecasting and predicting oil spills, oceanographic surveying of eddies and upwelling, or providing information on the distribution of biomass in ecosystems. We will be using the [Gulf Drifters Open dataset](https://zenodo.org/record/4421585), which contains all publicly available surface drifter trajectories from the Gulf of Mexico spanning 28 years.
# %%
from jax.config import config

config.update("jax_enable_x64", True)
from dataclasses import dataclass

from jax import hessian
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
import numpy as onp
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp
from tqdm import tqdm
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
colors = rcParams["axes.prop_cycle"].by_key()["color"]

# %%
# loading in data
try:
    gulf_data = pd.read_csv("data/gulfdatacoarse.csv")
except FileNotFoundError:
    gulf_data = pd.read_csv("HelmholtzGP/data/gulfdatacoarse.csv")
shape = (30, 30)

N = len(gulf_data["lonbins"])
pos = jnp.array([gulf_data["lonbins"], gulf_data["latbins"]]).T
vel = jnp.array([gulf_data["ubar"], gulf_data["vbar"]]).T
shape = (int(gulf_data["shape"][0]), int(gulf_data["shape"][1]))  # shape = (34,16)


pos_train = pos.T
vel_train = vel.T


# fig, ax = plt.subplots(1, 1)
# ax.quiver(
#     pos_train[0],
#     pos_train[1],
#     vel_train[0],
#     vel_train[1],
#     color=colors[1],
#     alpha=0.7,
#     label="$D_T$",
# )
# ax.legend()
# plt.show()

# Change vectors x -> X = (x,z), and vectors y -> Y = (y,z) via the artificial z label
def label_position(data):
    # introduce alternating z label
    n_points = len(data[0])
    label = jnp.tile(jnp.array([0.0, 1.0]), n_points)
    return jnp.vstack((jnp.repeat(data, repeats=2, axis=1), label)).T


def stack_velocity(data):
    return data.T.flatten().reshape(-1, 1)


pos_train3d = label_position(pos_train)
vel_train3d = stack_velocity(vel_train)

DT = gpx.Dataset(X=pos_train3d, y=vel_train3d)



def initialise_gp(kernel, mean, dataset):
    prior = gpx.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.Gaussian(
        num_datapoints=dataset.n, obs_noise=jnp.array([1.0], dtype=jnp.float64)
    )
    posterior = prior * likelihood
    return posterior





def optimise_mll(posterior, dataset, NIters=250, key=key, plot_history=True):
    # define the Marginal Log likelihood using DT
    objective = gpx.objectives.ConjugateMLL(negative=True)
    objective(posterior, train_data=DT)
    optimiser = ox.adam(learning_rate=0.1)
    # Optimise to minimise the MLL
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=objective,
        train_data=dataset,
        optim=optimiser,
        num_iters=NIters,
        safe=True,
        key=key,
    )
    if plot_history:
        fig, ax = plt.subplots(1, 1)
        ax.plot(history, color=colors[1])
        ax.set(xlabel="Training iteration", ylabel="Negative marginal log likelihood")

    return opt_posterior




def latent_distribution(opt_posterior, pos_3d):
    latent = opt_posterior.predict(pos_3d, train_data=DT)
    latent_mean = latent.mean()
    latent_std = latent.stddev()
    return latent_mean, latent_std



@dataclass
class helmholtz_kernel(gpx.kernels.AbstractKernel):
    # initialise Phi and Psi kernels as any stationary kernel in gpJax
    potetial_kernel = gpx.kernels.RBF(active_dims=[0, 1])
    stream_kernel = gpx.kernels.RBF(active_dims=[0, 1])

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # obtain indices for k_helm, implement in the correct sign between the derivatives
        z = jnp.array(X[2], dtype=int)
        zp = jnp.array(Xp[2], dtype=int)
        sign = (-1) ** (z + zp)
        # convert to array to correctly index, -ve sign due to exchange symmetry (only true for stationary kernels)
        potential_dvtve = -jnp.array(
            hessian(self.potetial_kernel)(X, Xp), dtype=jnp.float64
        )[z][zp]
        stream_dvtve = -jnp.array(
            hessian(self.stream_kernel)(X, Xp), dtype=jnp.float64
        )[1 - z][1 - zp]

        return potential_dvtve + sign * stream_dvtve

# Redefine Gaussian process with Helmholtz kernel
mean = gpx.mean_functions.Zero()
kernel = helmholtz_kernel()
helmholtz_posterior = initialise_gp(kernel, mean, DT)
# Optimise hyperparameters using optax

#opt_helmholtz_posterior = optimise_mll(helmholtz_posterior, DT)

opt_helmholtz_posterior = helmholtz_posterior

helmholtz_mean, helmholtz_std = latent_distribution(opt_helmholtz_posterior, pos_train3d)
vel_lat = [helmholtz_mean[::2].reshape(shape), helmholtz_mean[1::2].reshape(shape)]
pos_lat = pos_train


interval_lon = [-90.8,-83.8]
interval_lat = [24.0,27.5]



DeltaT = 0.8
trajectories_pos = []
trajectories_vel = []
ntrajs = 2
#nlocs = 15
everyother = 3
start_lons = [-90.0, -88.2, -84.8, -86.1, -90.5, -87.1, -90.27, -87.43]
start_lats = [ 26.5, 27.0, 26.0, 24.8, 27.5, 27.0, 27.3, 24.43,27.3]


for i in range(len(start_lons)):
    locs = []
    vels = []
    onp.random.seed(i)
    lon = onp.random.uniform(interval_lon[0], interval_lon[1])
    lat = onp.random.uniform(interval_lat[0], interval_lat[1])

    position = jnp.array([[start_lons[i]], [start_lats[i]]], dtype = jnp.float64)
    #locs.append(position.flatten())
    nlocs = onp.random.randint(4,8)
    for j in tqdm(range(nlocs)):
        if j%2 != 0:
            continue
        std = 0.01
        noise = onp.random.normal(0, std, 2)
        pos_labelled = label_position(position)
        velocity = opt_helmholtz_posterior.predict(pos_labelled, train_data=DT).mean()
        velocity = velocity + noise
        position = position.flatten() + (velocity) * DeltaT
        
        if j != nlocs-1:
            locs.append(position.flatten())
        vels.append(velocity.flatten())

        position = position.reshape(-1,1)

    locs = jnp.array(locs[::2]).T
    vels = jnp.array(vels[::2]).T

    trajectories_pos.append(locs)
    trajectories_vel.append(vels)





# make figure
fig, ax = plt.subplots(1,1)
fig.tight_layout()
ax.quiver(pos_lat[0], pos_lat[1], vel_lat[0], vel_lat[1], color = colors[0])
for i in range(len(start_lons)):
    ax.quiver(trajectories_pos[i][0], trajectories_pos[i][1], trajectories_vel[i][0], trajectories_vel[i][1], color = colors[i])
plt.show()

pos_tosave = jnp.hstack(trajectories_pos)
vel_tosave = jnp.hstack(trajectories_vel)

import pandas as pd
d = {'lon': pos_tosave[0], 'lat': pos_tosave[1], 'ubar': vel_tosave[0], 'vbar': vel_tosave[1]}
df = pd.DataFrame(data = d)
df.to_csv('gulfdata_train.csv')



print()

