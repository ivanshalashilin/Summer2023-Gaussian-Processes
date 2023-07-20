from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from jaxtyping import install_import_hook
import matplotlib.pyplot as plt
from simple_pytree import static_field
from dataclasses import dataclass
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
from jax.numpy import exp,abs,sin,cos


with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

key = jr.PRNGKey(5)

maxval = 2 * jnp.pi

#kernel, mean and prior

#attempt at making a periodic kernel but didn't really work ...
@dataclass
class PeriodicSE1D(gpx.kernels.AbstractKernel):
    Period: float = static_field(2.0*jnp.pi)
    Variance: float = static_field(1.0)
    Lengthscale: float = static_field(1.0)

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        Omega = 2 * jnp.pi/self.Period
        Ksin = self.Variance*exp(
            -0.5/self.Lengthscale**2 * abs(sin(x)-sin(y))
            )
            
        Kcos = self.Variance*exp(
            -0.5/self.Lengthscale**2 * abs(cos(x)-cos(y))
            )

        K = Kcos * Ksin
        return K.squeeze() 

#define kernel, mean and encode into prior
kernel = gpx.kernels.RBF() #PeriodicSE1D(Period = 0.01)
mean = gpx.mean_functions.Zero()
Prior = gpx.Prior(mean_function=mean, kernel=kernel)

#generate some noise data 
noise = 0.2
NPoints = 125
xMeasurements = jnp.linspace(0,maxval, NPoints).reshape(-1,1)
'''
Previously yTraining,yTesting etc. were labelled fTraining,fTesting. 
Strictly, the latter is incorrect because the points 
f represent the LATENT distrbution - the unobserved distribution 
that generated the data. 
y represent the PREDICTIVE distribution - the measurements made 
(ground truth), which have a gaussian likelihood, meaning they deviate 
normally from the latent function. 

A subtle but important point we will come back to later!
'''
yMeasurements = jnp.sin(xMeasurements) + noise*jr.normal(key = key, shape=xMeasurements.shape).reshape(-1,1) 

#split 80-20 training testing
from sklearn.model_selection import train_test_split
xTraining, xTesting, yTraining, yTesting = train_test_split(xMeasurements,yMeasurements, test_size=0.2, random_state=42)

#inputing data into gpjax, defining Gaussian likelihood and computing posterior
D = gpx.Dataset(X=xTraining, y=yTraining)
Likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=1.0)#setting 0 observation noise causes overfitting
Posterior = Prior * Likelihood

#Query points
NQueryPoints = 80
xQuery = jnp.linspace(0,2*jnp.pi, NQueryPoints).reshape(-1, 1)


LatentDistNoOpt = Posterior.predict(xQuery, train_data=D) # first time we have used the training data is here!! 
LatentMeanNoOpt = LatentDistNoOpt.mean().reshape(-1, 1)
LatentStdNoOpt = LatentDistNoOpt.stddev().reshape(-1, 1)

PredictiveDistNoOpt = Posterior.likelihood(LatentDistNoOpt)
PredictiveMeanNoOpt = PredictiveDistNoOpt.mean().reshape(-1, 1)
PredictiveStdNoOpt = PredictiveDistNoOpt.mean().reshape(-1, 1)

##UNOPTIMISED

#latent before optimisation
fig, ax = plt.subplots(1,1, figsize = (7.5,7))
ax.set_title('Predictive and Latent Distributions before Optimisation')
ax.plot(xQuery, jnp.sin(xQuery), color = 'blue', lw = 3, label = 'Underlying Function', alpha =0.8)
ax.plot(xTraining, yTraining, 'x', color = 'red', label = 'measurements')
ax.fill_between(
    xQuery.squeeze(),
    LatentMeanNoOpt.squeeze() - 2*LatentStdNoOpt.squeeze(),
    LatentMeanNoOpt.squeeze() + 2*LatentStdNoOpt.squeeze(),
    alpha=0.2,
    label="$\mu_{lat} \pm 2\sigma$",
    color='gray',
)

ax.plot(
    xQuery.squeeze(),
    LatentMeanNoOpt.squeeze(),
    alpha = 0.9,
    color = 'gray'
)

ax.plot(
    xQuery.squeeze(),
    PredictiveMeanNoOpt.squeeze() - 2*PredictiveStdNoOpt.squeeze(), 
    '--',
    alpha=1, label="$\mu_{pred} \pm 2\sigma$" ,
    color='gray',
)
ax.plot(
    xQuery.squeeze(),
    PredictiveMeanNoOpt.squeeze() + 2*PredictiveStdNoOpt.squeeze(), '--',
    alpha=1,
    color='gray',
)


ax.plot(xQuery, PredictiveMeanNoOpt, color = 'green', alpha = 0.7, label = '$\mu_{pred}$', lw = 5)
ax.plot(xQuery, LatentMeanNoOpt, color = 'red', alpha =0.7, label = '$\mu_{lat}$', lw = 1)
ax.legend()
plt.show()


#optimise parameters by optimising MLL (why marginal? only over training data?)
Objective = gpx.ConjugateMLL(negative=True)
Objective(Posterior, train_data=D)  # !!??
OptMeans = []
OptStd = []
Optimiser = ox.adam(learning_rate=0.01)
    
OptPosterior, history = gpx.fit(
    model=Posterior,
    objective=Objective,
    train_data=D,
    optim=Optimiser,
    num_iters=1000,
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

LatentDistOpt = OptPosterior.predict(xQuery, train_data=D)
LatentMeanOpt = LatentDistOpt.mean().reshape(-1, 1)
LatentStdOpt = LatentDistOpt.stddev().reshape(-1, 1)


PredictiveOpt = OptPosterior.likelihood(LatentDistOpt)
PredictiveMeanOpt = PredictiveOpt.mean().reshape(-1, 1)
PredictiveStdOpt = PredictiveOpt.stddev().reshape(-1,1)

#import
import matplotlib.pyplot as plt
#make figure
fig, ax = plt.subplots(1,1, figsize =(6,7))
ax.plot(xQuery, jnp.sin(xQuery), color = 'blue', lw = 3, label = 'Underlying Function', alpha =0.8)
ax.set_title('Predictive and Latent Distributions after Optimisation')

ax.plot(xTraining, yTraining, 'x', color = 'red', label = 'Measurmements') #or the groud truth


###LATENT DISTRIBUTION

ax.plot(
    xQuery.squeeze(), 
    LatentMeanOpt.squeeze(), 
    '--', color = 'green', 
    label = 'Optimised $\mu$'
    )


ax.fill_between(
    xQuery.squeeze(),
    LatentMeanOpt.squeeze() - 2*LatentStdOpt.squeeze(),
    LatentMeanOpt.squeeze() + 2*LatentStdOpt.squeeze(),
    alpha=0.2,
    label="$\mu_{lat} \pm 2\sigma$",
    color='gray',
)

## PREDICTIVE DISTRIBUTION
ax.plot(
    xQuery.squeeze(),
    PredictiveMeanOpt.squeeze(),
    alpha = 0.9,
    color = 'gray',
    label = '$\mu_{lat}$',
)

ax.plot(
    xQuery.squeeze(),
    PredictiveMeanOpt.squeeze() - 2*PredictiveStdOpt.squeeze(), 
    '--',
    alpha=1, label="$\mu_{pred} \pm 2\sigma$" ,
    color='gray',
)

ax.plot(
    xQuery.squeeze(),
    PredictiveMeanOpt.squeeze() + 2*PredictiveStdOpt.squeeze(), '--',
    alpha=1,
    color='gray',
)


# ax.plot(xQuery, PredictiveMeanOpt, color = 'green', alpha = 1, label = '$\mu_{pred}$', lw = 5)
# ax.plot(xQuery, PredictiveMeanOpt, color = 'red', alpha =1, label = '$\mu_{lat}$', lw = 1)
# # for i in range(int(NSamples/2-1)):
# #     ax.plot(xQuery.squeeze(),LatentSamples[i], color = 'red', alpha = 0.1)
# ax.plot(xQuery.squeeze(),LatentSamplesOpt[-1], color = 'red', alpha = 0.3, label = 'Latent Samples of $\mathbf{f}^\star$')

# # for i in range(int(NSamplesPredict/2-1)):
# #     ax.plot(xQuery[::2].squeeze(), PredictiveSample.squeeze()[i][::2], 'o', color = 'blue', alpha = 0.3)
# ax.plot(xQuery[::2].squeeze(), PredictiveSampleOpt.squeeze()[-1][::2], 'o', color = 'blue', label = 'Predictive Sample of $\mathbf{y}^\star$', alpha = 0.3)
ax.legend()
plt.show()

#having optimised the dataset, we can look at the 

#import
import matplotlib.pyplot as plt
fig, ax = plt.subplots(ncols=3, figsize=(9, 2.5), tight_layout=True)

#evaluate performance on testing points


LatentDistOptTest = OptPosterior.predict(xTesting, train_data=D)
PredictiveOpt = OptPosterior.likelihood(LatentDistOptTest)
PredictiveMeanOptTest = PredictiveOpt.mean().reshape(-1, 1)
PredictiveStdOptTest = PredictiveOpt.stddev().reshape(-1,1)

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

#compute rmse and r2: rmse should be close to 1, r2 close to 1
rmse = mean_squared_error(y_true=yTesting.squeeze(), y_pred=PredictiveMeanOptTest)
r2 = r2_score(y_true=yTesting.squeeze(), y_pred = PredictiveMeanOptTest)
print(f"Results:\n\tRMSE: {rmse: .4f}\n\t R2: {r2: .2f}") #\n newline, \t tab

Residuals = PredictiveMeanOptTest-yTesting

ax[0].scatter(PredictiveMeanOptTest, yTesting, color='red')
ax[0].plot([0, 1], [0, 1], transform=ax[0].transAxes)
ax[0].set(xlabel="Predicted", ylabel="Actual", title="Predicted vs Actual")

ax[1].scatter(PredictiveMeanOptTest.squeeze(), Residuals, color='red')
ax[1].plot([0, 1], [0.5, 0.5], transform=ax[1].transAxes) #
ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
ax[1].set_ylim([-1.0, 1.0])

import numpy as np
ax[2].hist(np.asarray(Residuals), bins=30, color='red')
ax[2].set_title("Residuals")



#ax.legend()
plt.show()

print()