from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt
import random
import math

intervals = []

for i in range(50):
    for j in range(5):
        intervall = []
        occurrences = 20
        for k in range(occurrences):
            intervall.append(1)
        intervals.append(intervall)
    for j in range(20):
        intervall = []
        occurrences = 5
        for k in range(occurrences):
            intervall.append(1)
        intervals.append(intervall)


plt.figure()
plt.plot([sum(count) for count in intervals])
plt.title("Person occurrences detected by robot before Poisson Process")
plt.show()
# Implement the Poisson model
alpha_t = 1.0
beta_t = 1.0
indicator = 1.0
lambda_t = alpha_t/beta_t
var_lamdba_t = alpha_t/(beta_t**2)
alphas = []
betas = []
lambdas = []
lambdas.append(lambda_t)
alphas.append(alpha_t)
betas.append(beta_t)

for i in range(1, len(intervals)):
    alpha_t = alpha_t + sum(intervals[i])*indicator
    beta_t = beta_t + 1*indicator
    lambda_t = float(alpha_t) / float(beta_t)
    alphas.append(alpha_t)
    betas.append(beta_t)
    lambdas.append(lambda_t)

print("Pimmelberger")
plt.figure()
plt.plot([sum(count) for count in intervals])
plt.title("Original detected occurrencies")
plt.plot(lambdas)
plt.title("Lamdba series calculated.")
plt.show()

# Now we need to plot the Poisson distributed likelihood
probability_gamma = []

#probability = gamma.pdf(x=i, a=alphas[4]+sum(intervals[4]), scale=betas[4]+sum(intervals[4]))
#probability = gamma.pdf(x=i, a=alphas[4], scale=betas[4])
#probability_gamma.append(probability)
#print("Pimmelberger")
#plt.figure()
#plt.plot(probability_gamma)
#plt.title("Probability distribution")
#plt.show()

x = np.arange(21)

plt.figure()
x = np.linspace(0, 40, num= 100)
plt.plot(x, gamma.pdf(x, a=alphas[3], scale=betas[3]))
plt.show()


