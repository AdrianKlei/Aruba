from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)

a = 1.99

#x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a), 100)
x = np.linspace(0, 20, num= 100)

ax.plot(x, gamma.pdf(x, a),
       'r-', lw=5, alpha=0.5, label='gamma pdf')
plt.show()