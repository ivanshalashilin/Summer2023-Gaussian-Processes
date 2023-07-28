
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

from numpy import genfromtxt
X_grid = genfromtxt('HelmholtzGP/data/Xgrid_vortex.csv', delimiter=',')
Y_grid = genfromtxt('HelmholtzGP/data/Ygrid_vortex.csv', delimiter=',')
XY_train = genfromtxt('HelmholtzGP/data/XY_train_vortex.csv', delimiter=',').T
XY_test = genfromtxt('HelmholtzGP/data/XY_test_vortex.csv', delimiter=',').T
UV_train = genfromtxt('HelmholtzGP/data/UV_train_vortex.csv', delimiter=',').T
UV_test = genfromtxt('HelmholtzGP/data/UV_test_vortex.csv', delimiter=',').T
div_grid = genfromtxt('HelmholtzGP/data/divgrid_vortex.csv', delimiter=',')
vort_grid = genfromtxt('HelmholtzGP/data/vortgrid_vortex.csv', delimiter=',')


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



#testdata 
xt = jnp.array([0.3,  0.4,  0.3,  0.4], dtype=jnp.float64)

SEVec = lambda x, V = jnp.array([1.0], dtype = jnp.float64),L = jnp.array([1.0], dtype = jnp.float64): V*exp(
                -0.5/L * ( (x[0]-x[2])**2 + (x[1]-x[3])**2 )
                )

hessian(SEVec)(xt)

@dataclass
class HelmholtzRBF(gpx.kernels.AbstractKernel):

    Var: Float[Array, "1 D"] = jnp.array([1.0,1.0])
    Lenscale: Float[Array, "1 D"]  = jnp.array([1.0,1.0])
    
    def __call__(
        self, x: Float[Array, "1 D"], xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
    

        SEVars = [x[0],x[1],xp[0],xp[1]]
        #psi and phi global parameters
        ParPhi = [self.Var[0], self.Lenscale[0]]
        ParPsi = [self.Var[1], self.Lenscale[1]]        
        
        def SEPhi(x):
            return ParPhi[0]**2 *exp(-0.5/ParPhi[1]**2 * ( (x[0]-x[2])**2 + (x[1]-x[3])**2 ))
        def SEPsi(x):
            return ParPsi[0]**2 *exp(-0.5/ParPsi[1]**2 * ( (x[0]-x[2])**2 + (x[1]-x[3])**2 ))
            

        #for debugging
        i = jnp.array(x[2], dtype = int)
        j = jnp.array(xp[2], dtype = int)

        #compute KPhi and KPsi Derivative via Hessian
        DivPhi =  -jnp.array(hessian(SEPhi)(SEVars),dtype = jnp.float64)[i][j]
        DivPsi =  -jnp.array(hessian(SEPsi)(SEVars),dtype = jnp.float64)[1-i][1-j]
        #DivsPsi =  -jnp.array(hessian(SEVec)(x,xp,*ParamsPhi))[x[2],xp[2]]

        Sign = (-1)**(x[2]+xp[2])
        
        Final = DivPhi + Sign * DivPsi
        
        return Final 

#initialise GP
D = gpx.Dataset(X=XYTrain3D, y = UVTrain3D)  
kernel = HelmholtzRBF()
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
Objective = gpx.ConjugateMLL(negative=True)
Objective(Posterior, train_data=D)  
Optimiser = ox.adam(learning_rate=0.01)


OptPosterior, history = gpx.fit(
    model=Posterior,
    objective=Objective,
    train_data=D,
    optim=Optimiser,
    num_iters=500,
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
ax[0].quiver(XY_test[0], XY_test[1], UV_test[0], UV_test[1]) #training data
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


