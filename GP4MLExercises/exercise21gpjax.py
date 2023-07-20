# Enable Float64 for more stable matrix inversions.
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox
import pandas as pd

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
import numpy as np
np.random.seed(2)
key = jr.PRNGKey(2)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


#define the kernel that describes the prior, encoding information about how we think the gaussian process should look. (Also set the prior mean to 0)
kernel = gpx.kernels.RBF(lengthscale =jnp.array(1.0), variance = jnp.array(1.0))
meanf = gpx.Zero()

#define the training data, which the prior will be conditioned on
xTraining = jnp.array([2.0,3.5,5.0,8.0]).reshape(-1, 1)
fTraining = 3*jnp.array([0.3,0.5,0.4,1.2]).reshape(-1, 1)
D = gpx.Dataset(X=xTraining, y=fTraining)

#choose a prior which is a gaussian process
prior = gpx.Prior(mean_function=meanf, kernel=kernel) 

#choose a gaussian likelihood (given the training data, how likely we were to observe the training points if we assume this likelihood is gaussian)
likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=0.3) #(we only used D.n (number of points {xi,yi} in gpx.Gaussian)
#units of obs_noise are [variance] ?

#compute the posterior, which we now know more about as we introduce training data
posterior = prior * likelihood 

# query the posterior at the test points given training data (our best knowledge so far given the data and assuming gaussian likelihood)
NPoints = 200
xQuery = jnp.linspace(0,10, NPoints).reshape(-1, 1)

#latent = unobserved. we never explicitly observe the conditioned mean
latent_dist = posterior.predict(xQuery, train_data=D) # first time we have used the training data is here!! 
latent_mean = latent_dist.mean() #at the points we want to query, given the training data
latent_std = latent_dist.stddev()

#predictive = samples of y star

# The predict method of a likelihood object transforms the latent distribution 
# of the Gaussian process. In the case of a Gaussian likelihood, this simply 
# applies the observational noise value to the diagonal values of the covariance matrix.
predictive_dist = posterior.likelihood(latent_dist)
predictive_mean = predictive_dist.mean()
predictive_std = predictive_dist.stddev()

#sample the posterior (candidate) functions at each point in our NPoints-D gaussian
NSamples = 3
LatentSamples = latent_dist.sample(seed=key, sample_shape=(NSamples,))

NSamplesPredict = 3
PredictiveSample = predictive_dist.sample(seed = key, sample_shape = (NSamplesPredict,))




#latent before optimisation
fig, ax = plt.subplots(1,1, figsize = (10,4))

ax.fill_between(
    xQuery.squeeze(),
    latent_mean.squeeze() - 2*latent_std.squeeze(),
    latent_mean.squeeze() + 2*latent_std.squeeze(),
    alpha=0.2,
    label="$\mu_{lat} \pm 2\sigma$",
    color='gray',
)

ax.plot(
    xQuery.squeeze(),
    latent_mean.squeeze(),
    alpha = 0.9,
    color = 'gray'
)

ax.plot(
    xQuery.squeeze(),
    predictive_mean.squeeze() - 2*predictive_std.squeeze(), 
    '--',
    alpha=1, label="$\mu_{pred} \pm 2\sigma$" ,
    color='gray',
)
ax.plot(
    xQuery.squeeze(),
    predictive_mean.squeeze() + 2*predictive_std.squeeze(), '--',
    alpha=1,
    color='gray',
)


ax.plot(xQuery, predictive_mean, color = 'green', alpha = 1, label = '$\mu_{pred}$', lw = 5)
ax.plot(xQuery, latent_mean, color = 'red', alpha =1, label = '$\mu_{lat}$', lw = 1)
# for i in range(int(NSamples/2-1)):
#     ax.plot(xQuery.squeeze(),LatentSamples[i], color = 'red', alpha = 0.1)
ax.plot(xQuery.squeeze(),LatentSamples[-1], color = 'red', alpha = 0.3, label = 'Latent Samples of $\mathbf{f}^\star$')

# for i in range(int(NSamplesPredict/2-1)):
#     ax.plot(xQuery[::2].squeeze(), PredictiveSample.squeeze()[i][::2], 'o', color = 'blue', alpha = 0.3)
ax.plot(xQuery[::2].squeeze(), PredictiveSample.squeeze()[-1][::2], 'o', color = 'blue', label = 'Predictive Sample of $\mathbf{y}^\star$', alpha = 0.3)


ax.legend()
plt.show()

#optimised posterior and associated optimised posterior likelihood, assuming the likelihood is gaussian (the points that generated the fit are normally distributed around the mean/sampled point)
from jax import jit
negative_mll = jit(gpx.ConjugateMLL(negative=True))
negative_mll(posterior, train_data = D) #¡¡¡!!!
opt_posterior, history = gpx.fit(
    model=posterior,
    objective=negative_mll,
    train_data=D,
    optim=ox.adam(learning_rate=0.01),
    num_iters=50000,
    safe=True,
    key=key,
)

#import
import matplotlib.pyplot as plt
#make figure
fig, ax = plt.subplots(1,1, figsize =(10,6))
ax.plot(history , 'x', color = 'red', label = '')
#plt.savefig('.png', dpi = 600)
plt.show()

latent_dist_opt = opt_posterior.predict(xQuery, train_data=D)
latent_mean_opt = latent_dist_opt.mean()
latent_std_opt = latent_dist_opt.stddev()

predictive_dist_opt = opt_posterior.likelihood(latent_dist_opt) #does it even make sense to consider an optimised predictive distribution??a?
predictive_mean_opt = predictive_dist_opt.mean()
predictive_std_opt = predictive_dist_opt.stddev()

NSamples = 3
LatentSamplesOpt = latent_dist_opt.sample(seed=key, sample_shape=(NSamples,))

NSamplesPredict = 3
PredictiveSampleOpt = predictive_dist_opt.sample(seed = key, sample_shape = (NSamplesPredict,))

# predictive and latent after optimisation
fig, ax = plt.subplots(1,1, figsize = (10,4))

ax.fill_between(
    xQuery.squeeze(),
    latent_mean_opt.squeeze() - 2*latent_std_opt.squeeze(),
    latent_mean_opt.squeeze() + 2*latent_std_opt.squeeze(),
    alpha=0.2,
    label="$\mu_{lat} \pm 2\sigma$",
    color='gray',
)

ax.plot(
    xQuery.squeeze(),
    latent_mean_opt.squeeze(),
    alpha = 0.9,
    color = 'gray'
)

ax.plot(
    xQuery.squeeze(),
    predictive_mean_opt.squeeze() - 2*predictive_std_opt.squeeze(), 
    '--',
    alpha=1, label="$\mu_{pred} \pm 2\sigma$" ,
    color='gray',
)
ax.plot(
    xQuery.squeeze(),
    predictive_mean_opt.squeeze() + 2*predictive_std_opt.squeeze(), '--',
    alpha=1,
    color='gray',
)




ax.plot(xQuery, predictive_mean_opt, color = 'green', alpha = 1, label = '$\mu_{pred}$', lw = 5)
ax.plot(xQuery, latent_mean_opt, color = 'red', alpha =1, label = '$\mu_{lat}$', lw = 1)
# for i in range(int(NSamples/2-1)):
#     ax.plot(xQuery.squeeze(),LatentSamples[i], color = 'red', alpha = 0.1)
ax.plot(xQuery.squeeze(),LatentSamplesOpt[-1], color = 'red', alpha = 0.3, label = 'Latent Samples of $\mathbf{f}^\star$')

# for i in range(int(NSamplesPredict/2-1)):
#     ax.plot(xQuery[::2].squeeze(), PredictiveSample.squeeze()[i][::2], 'o', color = 'blue', alpha = 0.3)
ax.plot(xQuery[::2].squeeze(), PredictiveSampleOpt.squeeze()[-1][::2], 'o', color = 'blue', label = 'Predictive Sample of $\mathbf{y}^\star$', alpha = 0.3)


ax.legend()
plt.show()