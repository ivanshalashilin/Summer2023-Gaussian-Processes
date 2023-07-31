# %% [markdown]
# # Gaussian Processes for Vector Fields
#
# In this notebook, we will go over an example of how to use GPJax for vector valued functions. 
# We will  how to create the relevant kernels and recreate the results from [Berlinghieri et al. (2023)](
#https://doi.org/10.48550/arXiv.2302.10364)
# create custom kernels.

# %% 
from jax.config import config
config.update("jax_enable_x64", True)
from simple_pytree import static_field
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
from jaxtyping import (Array,Float,Bool, install_import_hook,)
import optax as ox
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base.param import param_field
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
from dataclasses import dataclass
from jax.numpy import exp
from jax import hessian



#data

testname = 'duffing_divbig'

from numpy import genfromtxt
X_grid = genfromtxt('HelmholtzGP/data/Xgrid_'+testname+'.csv', delimiter=',')
Y_grid = genfromtxt('HelmholtzGP/data/Ygrid_'+testname+'.csv', delimiter=',')
XY_train = genfromtxt('HelmholtzGP/data/XY_train_'+testname+'.csv', delimiter=',').T
XY_test = genfromtxt('HelmholtzGP/data/XY_test_'+testname+'.csv', delimiter=',').T
UV_train = genfromtxt('HelmholtzGP/data/UV_train_'+testname+'.csv', delimiter=',').T
UV_test = genfromtxt('HelmholtzGP/data/UV_test_'+testname+'.csv', delimiter=',').T
div_grid = genfromtxt('HelmholtzGP/data/divgrid_'+testname+'.csv', delimiter=',')
vort_grid = genfromtxt('HelmholtzGP/data/vortgrid_'+testname+'.csv', delimiter=',')

gridsize = int(len(vort_grid))
NIters = 20000

#create R2nx3 dataset
NTrain = len(XY_train[0])
NTest = len(XY_test[0])
DimLabelTrain = jnp.tile(jnp.array([0.0,1.0]), NTrain)
DimLabelTest =  jnp.tile(jnp.array([0.0,1.0]), NTest)
XYTrain3D = jnp.vstack((jnp.repeat(XY_train, repeats =2 , axis = 1), DimLabelTrain)).T[0:2]
UVTrain3D = UV_train.T.flatten().reshape(-1,1)[0:2]
XYTest3D = jnp.vstack((jnp.repeat(XY_test, repeats =2 , axis = 1), DimLabelTest)).T[0:2]


#Helmholtz kernel

@dataclass
class HelmholtzRBF(gpx.kernels.AbstractKernel):

    Var: Float[Array, "1 D"] = jnp.array([1.0,1.0])
    Lenscale: Float[Array, "1 D"]  = jnp.array([1.0,1.0])
    Divergence: Bool = static_field(False)
    Vorticity: Bool = static_field(False)
    
    def __call__(
        self, x: Float[Array, "1 D"], xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
    
        #vector componentsof x and xprime
        SEVars = [x[0],x[1],xp[0],xp[1]]
        #psi and phi global parameters
        ParPhi = [self.Var[0], self.Lenscale[0]]
        ParPsi = [self.Var[1], self.Lenscale[1]]        
        #2D SE kernels with with associated hyperparameters
        def SEPhi(x):
            return self.kernel_1(x[0],x[1])
            return ParPhi[0]**2 * exp(-0.5/ParPhi[1]**2 * ( (x[0]-x[2])**2 + (x[1]-x[3])**2 ))
        def SEPsi(x):
            return ParPsi[0]**2 * exp(-0.5/ParPsi[1]**2 * ( (x[0]-x[2])**2 + (x[1]-x[3])**2 ))


        i = jnp.array(x[2], dtype = int)
        j = jnp.array(xp[2], dtype = int)
        #correct sign (±1)
        Sign = (-1)**(x[2]+xp[2])

        if self.Divergence:
            DivPhi4 = jnp.array(hessian(hessian(SEPhi))(SEVars))
            DivSumPhi = DivPhi4[0,0,0,0] + DivPhi4[0,2,0,2] + DivPhi4[2,0,2,0] + DivPhi4[2,2,2,2]

            DivPsi4 = jnp.array(hessian(hessian(SEPsi))(SEVars))
            DivSumPsi = DivPsi4[0,0,0,0] + DivPsi4[0,2,0,2] + DivPsi4[2,0,2,0] + DivPsi4[2,2,2,2]

            Final = DivSumPhi + Sign * DivPsi4
            return Final

        #necessary KPhi and KPsi 2nd derivatives via Hessian
        #-1 because of exchange symmetry between x and xp when computing x and xp
        DivPhi =  -jnp.array(hessian(SEPhi)(SEVars),dtype = jnp.float64)[i][j]
        DivPsi =  -jnp.array(hessian(SEPsi)(SEVars),dtype = jnp.float64)[1-i][1-j] 
        
        return DivPhi + Sign * DivPsi

#initialise GP
D = gpx.Dataset(X=XYTrain3D, y = UVTrain3D)  
kernel = HelmholtzRBF(Divergence = False)
mean = gpx.mean_functions.Zero()
Prior = gpx.Prior(mean_function=mean, kernel=kernel)
Likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=jnp.array([0.02392433],dtype=jnp.float64))
Posterior = Prior*Likelihood

#unoptimised posterior
LatentDist = Posterior.predict(XYTest3D, train_data=D)
LatentMean = LatentDist.mean()#.reshape(-1, 1)
LatentStd = LatentDist.stddev()#.reshape(-1, 1)

UTestLat = LatentMean[::2]
VTestLat = LatentMean[1::2]


#optimise hyperparameters by extremising MLL 
Objective = gpx.ConjugateMLL(negative=True)
Objective(Posterior, train_data=D)  
Optimiser = ox.adam(learning_rate=0.01)


OptPosterior, history = gpx.fit(
    model=Posterior,
    objective=Objective,
    train_data=D,
    optim=Optimiser,
    num_iters=NIters,
    safe = True,
    key=key,
)


fig, ax = plt.subplots(1,1, figsize =(10,6))
ax.plot(history)
plt.show()

LatentDistOpt = OptPosterior.predict(XYTest3D, train_data=D)
LatentMeanOpt = LatentDistOpt.mean()
LatentStdOpt = LatentDistOpt.stddev()

UTestOpt = LatentMeanOpt[::2]
VTestOpt = LatentMeanOpt[1::2]


#plot ground truth, unoptimised and optimised latent distribution
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].set(title = 'Ground Truth')
ax[0].quiver(XY_test[0], XY_test[1], UV_test[0], UV_test[1]) #training data
ax[0].quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') 
ax[0].set_aspect('equal')

ax[1].set(title = 'Unoptimised LD')
ax[1].quiver(XY_test[0], XY_test[1], UTestLat, VTestLat) #training data
ax[1].quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') 
ax[1].set_aspect('equal')

ax[2].set(title = 'Optimised LD')
ax[2].quiver(XY_test[0].reshape(gridsize,gridsize), XY_test[1].reshape(gridsize,gridsize), UTestOpt.reshape(gridsize,gridsize), VTestOpt.reshape(gridsize,gridsize)) #training data
ax[2].quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'red') #testing data
ax[2].set_aspect('equal')
fig.tight_layout()
plt.savefig('HelmholtzGP/plots/'+testname+'png',dpi = 600)
plt.show()

#compare Helmholtz GP to ground truth by 
X = XY_test[0]
Y = XY_test[1]
fig, ax = plt.subplots(1,1, figsize =(8.5,6))
fig.tight_layout()
Residuals =jnp.sqrt((UV_test[0]-UTestOpt)**2 + (VTestOpt-UV_test[1])**2)
m = ax.imshow(Residuals.reshape(gridsize,gridsize), extent=[X.min(), X.max(), Y.min(), Y.max()],cmap = 'hot')
plt.colorbar(m, cmap = 'hot') 
#plt.savefig('.png', dpi = 600)
ax.quiver(XY_train[0],XY_train[1], UV_train[0], UV_train[1], color = 'green') #testing data
plt.savefig('HelmholtzGP/plots/'+testname+'residuals.png', dpi = 600)
plt.show()


#comparing optimised results to those of the paper by comparing residuals 
import pickle
pickle_in = open('.pkl', 'rb')
import torch
UVPaper = pickle.load(pickle_in)
UVPaper = jnp.array(UVPaper[0].detach().numpy()).reshape(-1,1)

#sort in ascending order and observe that residuals are small
PaperResiduals = (jnp.sort(UVPaper, axis = 0)-jnp.sort(LatentMeanOpt.reshape(-1,1), axis = 0)).flatten()


#make figure
fig, ax = plt.subplots()
ax.hist(PaperResiduals, bins = 15)
ax.set(xlabel = 'Residuals', title = 'Residuals between GPJax and Publication')
#plt.savefig('.png', dpi = 600)
plt.show()





