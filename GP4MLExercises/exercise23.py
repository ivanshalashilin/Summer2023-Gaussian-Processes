import numpy as np
from numpy.linalg import inv
from numpy import matmul 
import matplotlib.pyplot as plt
np.random.seed(0)
'''
The Wiener process is defined for x ≥ 0 and has f(0)=0. (See section B.2.1 for further details.) 
It has mean zero and a non-stationary covariance function k(x, x')= min(x, x'). If we condition
on the Wiener process passing through f(1)=0 we obtain a process known as the Brownian bridge 
(or tied-down Wiener process). Show that this process has covariance k(x, x')=\min (x, x')-x.x' 
for 0 ≤ x, x' ≤ 1 and mean 0 . Write a computer program to draw samples from this process at 
a finite grid of x points in [0,1].
'''

def SquareKernel(xInput, error = None):#unsure if this is the right nomenclature
    #xp and xq some set of points in the input space, but for the exercise should be scalars
    Dim = len(xInput)
    xRowmat = np.tile(xInput,(Dim,1))
    xColmat = xRowmat.T

    if error is None:
        return np.minimum(xRowmat,xColmat)-xRowmat*xColmat
    ErrorMat = error**2 * np.identity(Dim)
    return np.minimum(np.minimum(xRowmat,xColmat)-xRowmat*xColmat) + ErrorMat

def RectKernel(xTraining, xQuery):

    #don't think this is needed? but constructs if needed

    #choose to compute k(X,X*), then tranpose to get other corner
    #X* are the points we want to query, X are the training data
    #offdiagonal submatrix 
    DimQuery = len(xQuery)
    DimTraining = len(xTraining)

    xTrainingQueryDim = np.tile(xTraining,(DimQuery,1))
    xQueryTrainingDim = np.tile(xQuery, (DimTraining,1))

    # X-X*
    

    return np.minimum(xQueryTrainingDim,xTrainingQueryDim.T)-xQueryTrainingDim*xTrainingQueryDim.T


def SamplePosterior(xTraining,xQuery,fTraining):

    #Training Data Kernel (top left)
    kXX = SquareKernel(xTraining)
    #Query Data Kernel (bottom right)
    kXStarXStar = SquareKernel(xQuery)
    #Off Diagonal (Top Right)
    kXXStar = RectKernel(xTraining,xQuery) #top right
    kXStarX = kXXStar.T #bottom left

    #construct the normal distribution to sample (equation 2.19)

    muConditioned = matmul(matmul(kXStarX,inv(kXX)),fTraining)
    covConditioned = kXStarXStar-matmul(kXStarX,matmul(inv(kXX), kXXStar))

    #sample the conditioned N dimensional Gaussian (MVG) distribution
    GaussianProcess = np.random.multivariate_normal(muConditioned,covConditioned)

    return GaussianProcess, muConditioned, covConditioned



#use a regular grid of scalar inputs 0≤x≤1
NPoints = 400
xQuery = np.linspace(0.0,1.0,NPoints)

#GP without training data
cov = SquareKernel(xQuery)
mu = np.zeros(NPoints)
#sample the gaussian, and get the y values
#plot GPs only given the kernel
fig,ax = plt.subplots(1,1, figsize =(10,6))
#generate and plot GPs
for i in range(3):
    fQuery = np.random.multivariate_normal(mu,cov)
    ax.plot(xQuery, fQuery, color = 'red', alpha = 0.4)
fQuery = np.random.multivariate_normal(mu,cov) 
ax.plot(xQuery, fQuery, color = 'red', label = 'GP', alpha = 0.3)
#use marginalisation(?) to get the pm 2 sigma range
MuPlusTwoSigma = mu+2*np.sqrt(np.diagonal(cov))
MuMinusTwoSigma = mu-2*np.sqrt(np.diagonal(cov))
ax.fill_between(xQuery, MuPlusTwoSigma,MuMinusTwoSigma, color = 'gray', alpha = 0.3, label = '$\mu \pm 2 \sigma$')
ax.legend()
plt.show()


####conditioned GP given training data#####

#training data (don't think it is classed as part of the prior? uncertain)
xTraining = np.array([0.3,0.35, 0.4,0.5,0.6,0.8])
fTraining = np.array([0.5,-0.2,0.1,0.9,0.8,-0.1])

xTraining = np.linspace(0.01,0.99,10)
fTraining = 2*xTraining+3

#plot
fig, ax = plt.subplots(1,1, figsize =(10,6))
#generate and plot *conditioned* GPs *given* the training data
for i in range(3):
    fQueryGivenTraining, muConditioned, covConditioned = SamplePosterior(xTraining,xQuery,fTraining)
    ax.plot(xQuery, fQueryGivenTraining, color = 'red', alpha = 0.4)

fQueryGivenTraining, muConditioned, covConditioned = SamplePosterior(xTraining,xQuery,fTraining)
ax.plot(xQuery, fQueryGivenTraining, color = 'red', label = 'GP', alpha = 0.3)
#use marginalisation(?) to get the pm 2 sigma range
MuPlusTwoSigma = muConditioned+2*np.sqrt(np.diagonal(covConditioned))
MuMinusTwoSigma = muConditioned-2*np.sqrt(np.diagonal(covConditioned))
ax.fill_between(xQuery, MuPlusTwoSigma,MuMinusTwoSigma, color = 'gray', alpha = 0.3, label = '$\mu \pm 2 \sigma$')
#plot the training data
ax.plot(xTraining, fTraining, '+' ,color = 'black', label = 'Training Data', ms = 20)
ax.legend()
plt.show()

