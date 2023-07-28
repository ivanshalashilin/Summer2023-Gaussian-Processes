
# Gaussian Processes for Vector Fields



from jax.config import config

config.update("jax_enable_x64", True)

import torch
torch.manual_seed(123)
from jax import jit
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax as ox
from jax import grad
from jax.config import config

from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base.param import param_field


# Enable Float64 for more stable matrix inversions.
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

from simple_pytree import static_field
from dataclasses import dataclass

#import data

X_grid = np.genfromtxt('HelmholtzGP/data/Xgrid_vortex.csv', delimiter=',')
Y_grid = np.genfromtxt('HelmholtzGP/data/Ygrid_vortex.csv', delimiter=',')
XY_train = np.genfromtxt('HelmholtzGP/data/XY_train_vortex.csv', delimiter=',').T
XY_test = np.genfromtxt('HelmholtzGP/data/XY_test_vortex.csv', delimiter=',').T
UV_train = np.genfromtxt('HelmholtzGP/data/UV_train_vortex.csv', delimiter=',').T
UV_test = np.genfromtxt('HelmholtzGP/data/UV_test_vortex.csv', delimiter=',').T
div_grid = np.genfromtxt('HelmholtzGP/data/divgrid_vortex.csv', delimiter=',')
vort_grid = np.genfromtxt('HelmholtzGP/data/vortgrid_vortex.csv', delimiter=',')


#make figure
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


# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,1, figsize =(6,6))
# ax.quiver(XY_test[0], XY_test[1], UV_test[0], UV_test[1]) #training data
# ax.quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') #testing data
# ax.set_aspect('equal')
# ax.set(xlim = [-1.3,1.3], ylim = [-1.3,1.3])
# plt.show()

#UVTrain = 


y = jnp.array([[0.3,  0.5,  0.        ],
       [0.3,  0.5,  1.        ],
       [0.3, 0.5,  0.        ],
       [0.3, 0.5,  1.        ]], dtype=jnp.float64)


x = jnp.array([[0.3,  0.5,  0.        ],
       [0.3,  0.5,  1.        ],
       [0.3, 0.5,  0.        ],
       [0.3, 0.5,  1.        ]], dtype=jnp.float64)

from jax.numpy import exp
@dataclass
class VelocityRBF(gpx.kernels.AbstractKernel):
    Var: Float[Array, "1 D "] = param_field(jnp.array([1.0,1.0]))
    Lenscale: Float[Array, "1 D"]  = param_field(jnp.array([1.0,1.0]))

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        #standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0
    
        Kexp = lambda xl, yl, var, ls: var*exp(
                -0.5/ls**2 * abs(xl-yl)
                )

        KexpX = Kexp(x[0],y[0], self.Var[0], self.Lenscale[0])
        KexpY = Kexp(x[1],y[1], self.Var[1], self.Lenscale[1])
        
        Switch = (y[2]+x[2]+1)%2
        
        return KexpX * KexpY * Switch

D = gpx.Dataset(X=XYTrain3D, y = UVTrain3D)      
#D = gpx.Dataset(X=x, y = y)      
kernel = VelocityRBF()
mean = gpx.mean_functions.Zero()
Prior = gpx.Prior(mean_function=mean, kernel=kernel)

Likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=0.13)

Posterior = Prior*Likelihood

LatentDistNoOpt = Posterior.predict(XYTest3D, train_data=D)
LatentMeanNoOpt = LatentDistNoOpt.mean()#.reshape(-1, 1)
LatentStdNoOpt = LatentDistNoOpt.stddev()#.reshape(-1, 1)

UTest = LatentMeanNoOpt[::2]
VTest = LatentMeanNoOpt[1::2]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(6,6))
ax.set(title = 'Unoptimised Latent Distribution')
ax.quiver(XY_test[0], XY_test[1], UTest, VTest) #training data
ax.quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') #testing data
ax.set_aspect('equal')
ax.set(xlim = [-1.3,1.3], ylim = [-1.3,1.3])
plt.show()


#optimise parameters by optimising MLL (why marginal? only over training data?)
Objective = gpx.ConjugateMLL(negative=True)
Objective(Posterior, train_data=D)  # !!??
Optimiser = ox.adam(learning_rate=0.01)
    
OptPosterior, history = gpx.fit(
    model=Posterior,
    objective=Objective,
    train_data=D,
    optim=Optimiser,
    num_iters=50000,
    safe = True,
    key=key,
)

#import
import matplotlib.pyplot as plt
#make figure
fig, ax = plt.subplots(1,1, figsize =(10,6))
ax.plot(history, color = 'red', label = '')
#plt.savefig('.png', dpi = 600)
plt.show()

LatentDistOpt = OptPosterior.predict(XYTest3D, train_data=D)
LatentMeanOpt = LatentDistOpt.mean()#.reshape(-1, 1)
LatentStdOpt = LatentDistOpt.stddev()#.reshape(-1, 1)

UTestOpt = LatentMeanOpt[::2]
VTestOpt = LatentMeanOpt[1::2]


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(6,6))
ax.set(title = 'Optimised Latent Distribution')
ax.quiver(XY_test[0].reshape(17,17), XY_test[1].reshape(17,17), UTestOpt.reshape(17,17), VTestOpt.reshape(17,17)) #training data
ax.quiver(XY_train[0],XY_train[1], UV_train[1], UV_train[0], color = 'red') #testing data
ax.set_aspect('equal')
ax.set(xlim = [-1.3,1.3], ylim = [-1.3,1.3])
plt.show()

Residuals = np.sqrt((UTest-UTestOpt)**2 + (VTest-VTestOpt)**2)
#import
import matplotlib.pyplot as plt
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

print()