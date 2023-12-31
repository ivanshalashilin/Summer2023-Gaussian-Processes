import numpy as np
from numpy.linalg import inv
from numpy import matmul 
import matplotlib.pyplot as plt
np.random.seed(2)
from tqdm import tqdm


'''
1. Replicate the generation of random functions from Figure 2.2 . Use a regular (or random) 
grid of scalar inputs and the covariance function from eq. (2.16). Hints on how to generate 
random samples from multi-variate Gaussian distributions are given in section A.2. Invent 
some training data points, and make random draws from the resulting GP posterior using eq. (2.19).
'''

def SquareKernel(xInput, error = None):#unsure if this is the right nomenclature
    #xp and xq some set of points in the input space, but for the exercise should be scalars
    Dim = len(xInput)
    xRowmat = np.tile(xInput,(Dim,1))

    DifferenceMat = xRowmat-xRowmat.T    
   
    if error is None:
        return np.exp(-0.5*np.abs(DifferenceMat)**2)
    ErrorMat = error**2 * np.identity(Dim)
    return np.exp(-0.5*np.abs(DifferenceMat)**2) + ErrorMat

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
    DifferenceMat = xTrainingQueryDim.T-xQueryTrainingDim

    return np.exp(-0.5*np.abs(DifferenceMat)**2)


def SamplePosterior(xTraining,xQuery,fTraining, error = None):

    #Training Data Kernel (top left with a small error)
    kXX = SquareKernel(xTraining, error)
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



#use a regular grid of scalar inputs
NPoints = 400
xQuery = np.linspace(0,10,NPoints)

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
xTraining = np.array([2,3.5,5,8])
fTraining = 3*np.array([0.3,0.5,0.4,1.2])
muTraining = fTraining
error = np.sqrt(10.0) #small uncertainty on the training data
#plot
fig, ax = plt.subplots(1,1, figsize =(10,6))
#generate and plot *conditioned* GPs *given* the training data
NSamples = 3
for i in tqdm(range(NSamples-1)):
    fQueryGivenTraining, muConditioned, covConditioned = SamplePosterior(xTraining,xQuery,fTraining, error)
    ax.plot(xQuery, fQueryGivenTraining, color = 'red', alpha = 0.4)
fQueryGivenTraining, muConditioned, covConditioned = SamplePosterior(xTraining,xQuery,fTraining,error)
ax.plot(xQuery, fQueryGivenTraining, color = 'red', label = 'GP', alpha = 0.3)
#use marginalisation(?) to get the pm 2 sigma range
MuPlusTwoSigma = muConditioned+2*np.sqrt(np.diagonal(covConditioned))#+2*error
MuMinusTwoSigma = muConditioned-2*np.sqrt(np.diagonal(covConditioned))#-2*error
ax.fill_between(xQuery, MuPlusTwoSigma,MuMinusTwoSigma, color = 'gray', alpha = 0.3, label = '$\mu \pm 2 \sigma$')
#plot the training data
ax.plot(xTraining, fTraining,'+' , ms = 20)

ax.plot(xQuery, muConditioned, color = 'gray', alpha =0.8)

#ax.errorbar(xTraining, fTraining,yerr=2*error, marker = '+' , ls = 'none' ,color = 'black', label = 'Training Data', ms = 20)
ax.legend()
plt.show()

