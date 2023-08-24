
# Gaussian Processes for Vector Fields
from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib import rcParams
import jax.numpy as jnp
import optax as ox
import tensorflow_probability as tfp
from jax.config import config
from dataclasses import dataclass
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
from jax import hessian
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = rcParams["axes.prop_cycle"].by_key()["color"]

#import data
import numpy as onp
XY_train =onp.genfromtxt('HelmholtzGP/data/XY_train_vortex.csv', delimiter=',').T
XY_test =onp.genfromtxt('HelmholtzGP/data/XY_test_vortex.csv', delimiter=',').T
UV_train =onp.genfromtxt('HelmholtzGP/data/UV_train_vortex.csv', delimiter=',').T
UV_test =onp.genfromtxt('HelmholtzGP/data/UV_test_vortex.csv', delimiter=',').T

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,1, figsize =(6,6))
# ax.quiver(XY_test[0], XY_test[1], UV_test[0], UV_test[1]) #training data
# ax.quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') #testing data
# ax.set_aspect('equal')
# ax.set(xlim = [-1.3,1.3], ylim = [-1.3,1.3])
# plt.show()

#create R2nx3 dataset
NTrain = len(XY_train[0])
NTest = len(XY_test[0])
DimLabelTrain = jnp.tile(jnp.array([0.0,1.0]), NTrain)
DimLabelTest =  jnp.tile(jnp.array([0.0,1.0]), NTest)
XYTrain3D = jnp.vstack((jnp.repeat(XY_train, repeats =2 , axis = 1), DimLabelTrain)).T
UVTrain3D = UV_train.T.flatten().reshape(-1,1)
XYTest3D = jnp.vstack((jnp.repeat(XY_test, repeats =2 , axis = 1), DimLabelTest)).T


@dataclass
class Helmholtz(gpx.kernels.AbstractKernel):

    kernelPhi: gpx.kernels.stationary = gpx.kernels.RBF(active_dims = [0,1])
    kernelPsi: gpx.kernels.stationary = gpx.kernels.RBF(active_dims = [0,1])
    
    def __call__(
        self, x: Float[Array, "1 D"], xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:

        i = jnp.array(x[2], dtype = int)
        j = jnp.array(xp[2], dtype = int)
        Sign = (-1)**(i+j)
        DivPhi =  -jnp.array(hessian(self.kernelPhi)(x,xp),dtype = jnp.float64)[i][j]
        DivPsi =  -jnp.array(hessian(self.kernelPsi)(x,xp),dtype = jnp.float64)[1-i][1-j]

        return DivPhi + Sign * DivPsi


#initialise GP
D = gpx.Dataset(X=XYTrain3D, y = UVTrain3D)  
kernel = Helmholtz()
mean = gpx.mean_functions.Zero()
Prior = gpx.Prior(mean_function=mean, kernel=kernel)
Likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=jnp.array([0.02392433],dtype=jnp.float64))
Posterior = Prior*Likelihood

LatentDist = Posterior.predict(XYTest3D, train_data=D)
LatentMean = LatentDist.mean()#.reshape(-1, 1)
LatentStd = LatentDist.stddev()#.reshape(-1, 1)

UTestLat = LatentMean[::2]
VTestLat = LatentMean[1::2]

#optimise parameters by optimising MLL 
Objective = gpx.objectives.ConjugateMLL(negative=True)
Objective(Posterior, train_data=D)  
Optimiser = ox.adam(learning_rate=0.01)

NIters = 500
OptPosterior, history = gpx.fit(
    model=Posterior,
    objective=Objective,
    train_data=D,
    optim=Optimiser,
    num_iters=NIters,
    safe = True,
    key=key,
)

LatentDistOpt = OptPosterior.predict(XYTest3D, train_data=D)
LatentMeanOpt = LatentDistOpt.mean()#.reshape(-1, 1)
LatentStdOpt = LatentDistOpt.stddev()#.reshape(-1, 1)

UTestOpt = LatentMeanOpt[::2]
VTestOpt = LatentMeanOpt[1::2]


import matplotlib.pyplot as plt
#make figure
fig, ax = plt.subplots(1,1, figsize =(10,6))
ax.plot(history, color = 'red', label = '')
#plt.savefig('.png', dpi = 600)
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3, figsize = (12,6))
ax[0].set(title = 'Ground Truth')
ax[0].quiver(XY_test[0], XY_test[1], UV_test[0], UV_test[1]) 
ax[0].quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') 
ax[0].set_aspect('equal')
ax[0].set(xlim = [-1.3,1.3], ylim = [-1.3,1.3])


ax[1].set(title = 'Unoptimised LD')
ax[1].quiver(XY_test[0], XY_test[1], UTestLat, VTestLat) #training data
ax[1].quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') 
ax[1].set_aspect('equal')
ax[1].set(xlim = [-1.3,1.3], ylim = [-1.3,1.3])

ax[2].set(title = 'Optimised LD')
ax[2].quiver(XY_test[0].reshape(17,17), XY_test[1].reshape(17,17), UTestOpt.reshape(17,17), VTestOpt.reshape(17,17)) #training data
ax[2].quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') #testing data
ax[2].set_aspect('equal')
ax[2].set(xlim = [-1.3,1.3], ylim = [-1.3,1.3])
plt.show()


#import
import pickle
pickle_in = open('.pkl', 'rb')
import torch
UVPaper = pickle.load(pickle_in)
UVPaper = jnp.array(UVPaper[0].detach().numpy()).reshape(-1,1)

#comparing all results in paper, residuals are small, so results are in agreement
PaperResiduals = jnp.sort(UVPaper, axis = 0)-jnp.sort(LatentMeanOpt.reshape(-1,1), axis = 0)
Residuals =jnp.sqrt((UV_test[0]-UTestOpt)**2 + (VTestOpt-UV_test[1])**2)
#make figure
X = XY_test[0]
Y = XY_test[1]
fig, ax = plt.subplots(1,1, figsize =(10,6))
fig.tight_layout()
m = ax.imshow(Residuals.reshape(17,17), 
extent=[X.min(), X.max(), Y.min(), Y.max()],
 cmap = 'hot')
plt.colorbar(m, cmap = 'hot') 
#plt.savefig('.png', dpi = 600)
ax.quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'green') #testing data
plt.show()

#NLPD 

NormalDist = tfp.substrates.jax.distributions.Normal(
    loc = LatentMeanOpt, # [N, 1]
    scale = jnp.sqrt(LatentStdOpt**2), # [N, 1]
)

