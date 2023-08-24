
# Gaussian Processes for Vector Fields


'''
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
import numpy as jnp
import optax as ox
from jax import grad
from jax.config import config
from tqdm import tqdm
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)



with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

from simple_pytree import static_field
from dataclasses import dataclass

#import data

X_grid =jnp.genfromtxt('HelmholtzGP/data/Xgrid_vortex.csv', delimiter=',')
Y_grid =jnp.genfromtxt('HelmholtzGP/data/Ygrid_vortex.csv', delimiter=',')
XY_train =jnp.genfromtxt('HelmholtzGP/data/XY_train_vortex.csv', delimiter=',').T
XY_test =jnp.genfromtxt('HelmholtzGP/data/XY_test_vortex.csv', delimiter=',').T
UV_train =jnp.genfromtxt('HelmholtzGP/data/UV_train_vortex.csv', delimiter=',').T
UV_test =jnp.genfromtxt('HelmholtzGP/data/UV_test_vortex.csv', delimiter=',').T
div_grid =jnp.genfromtxt('HelmholtzGP/data/divgrid_vortex.csv', delimiter=',')
vort_grid =jnp.genfromtxt('HelmholtzGP/data/vortgrid_vortex.csv', delimiter=',')




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
'''
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
from jaxtyping import (Array,Float,install_import_hook,)
import optax as ox
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
from simple_pytree import static_field
from dataclasses import dataclass
from jax.numpy import exp, hstack as stk
from jax import  vmap, hessian, device_put

#toy data
x = jnp.array([[-0.89999998,  0.89999998,  0.        ],
       [-0.89999998,  0.89999998,  1.        ],
       [-0.80000001, -0.89999998,  0.        ],
       [-0.80000001, -0.89999998,  1.        ]], dtype=jnp.float64)

y = jnp.array([[-0.89999998,  0.89999998,  0.        ],
       [-0.89999998,  0.89999998,  1.        ],
       [-0.80000001, -0.89999998,  0.        ],
       [-0.80000001, -0.89999998,  1.        ]], dtype=jnp.float64)

@dataclass
class HelmholtzRBF(gpx.kernels.AbstractKernel):
    Var: float = static_field(jnp.array([1.0,0.5], dtype=jnp.float64))
    Lenscale: float = static_field(jnp.array([1.0,0.5],dtype=jnp.float64))
    

    def __call__(
        self, x: Float[Array, "1 D"], xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        
        #define pairwise SE
        def SE(x1, x2, xp1, xp2, V=1 , L=1 ): 
            return V*exp( -0.5/L * ( (x1-xp1)**2 + (x2-xp2)**2 ) )

        ParamsPhi = [self.Var[0], self.Lenscale[0]]
        ParamsPsi = [self.Var[1], self.Lenscale[1]]

        #get each vecotr component, store in List
        x1 = jnp.array(x.val.T[0][::2], dtype = jnp.float64)
        x2 = jnp.array(x.val.T[1][::2], dtype = jnp.float64)
        xp1 = jnp.array(xp.val.T[0][::2], dtype = jnp.float64)
        xp2 = jnp.array(xp.val.T[1][::2], dtype = jnp.float64)
        SEVars = [x1,x2,xp1,xp2]
        
        #for debugging
        # SEVars = [
        #     jnp.array([0.3,0.3,0.3]),
        #     jnp.array([0.5,0.5,0.5]),
        #     jnp.array([0.4,0.4,0.4]),
        #     jnp.array([1.0,1.0,1.0]),
        #     ]

        #compute KPhi and KPsi Derivative via Hessian, put into jnp array
        HessPhi = hessian((SE), [0,1])
        vmHessPhi = vmap(
            vmap(HessPhi, 
            in_axes = (0,None,0,None,None,None)),
            in_axes = (None,0,None,0,None,None)
            )
        HessPsi = hessian((SE), [1,0])
        vmHessPsi = vmap(
            vmap(HessPsi, 
            in_axes = (0,None,0,None,None,None)), 
            in_axes = (None,0,None,0,None,None)
            )
        #evaluation and array manipulation to get correct shape
        DivsPhi = stk(stk(-jnp.array(vmHessPhi(*SEVars, *ParamsPhi), dtype = jnp.float64).T),dtype = jnp.float64)
        DivsPsi = stk(stk(-jnp.array(vmHessPsi(*SEVars, *ParamsPsi), dtype = jnp.float64).T), dtype=jnp.float64)

        #Correct alternating sign 
        Sign = jnp.array((-1)**(x[2]+xp[2]).val.val, dtype=jnp.float64)
        
        #output
        Final = jnp.array(DivsPhi + Sign * DivsPsi)

        return Final

#initialise GP
D = gpx.Dataset(X=x, y = y)  
kernel = HelmholtzRBF()
mean = gpx.mean_functions.Zero()
Prior = gpx.Prior(mean_function=mean, kernel=kernel)
Likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=jnp.array([0.13],dtype=jnp.float64))
Posterior = Prior*Likelihood

#optimise parameters by optimising MLL 
Objective = gpx.ConjugateMLL(negative=True)
Objective(Posterior, train_data=D)  # <--- error here
Optimiser = ox.adam(learning_rate=0.01)


'''
    
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
ax.quiver(XY_test[0].reshape(17,17), XY_test[1].reshape(17,17), UTestOpt.reshape(17,17), VTestOpt.reshape(17,17)) #training data
ax.quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') #testing data
ax.set_aspect('equal')
ax.set(xlim = [-1.3,1.3], ylim = [-1.3,1.3])
plt.show()

Residuals =jnp.sqrt((UV_test[0]-UTestOpt)**2 + (VTestOpt-UV_test[1])**2)
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


'''