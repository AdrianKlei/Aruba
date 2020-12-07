# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Set script-wide font size
plt.rc("font", size=11, family="Helvetica")
params = {'legend.fontsize': 9 }
plt.rcParams.update(params)

# Figure and panel objects
fig, panels = plt.subplots(5, 1, sharex=True)

# Plotted data
x     = np.arange(0, 100, 0.1)
y_sin = np.sin(x)*100
y_cos = np.cos(x)
y_cos_exp = np.cos(x) * np.exp(-x / 20.0)

# Labels
panels[0].set_title("Title")
panels[-1].set_xlabel("X label")

panels[0].set_ylabel("Sin")
panels[1].set_ylabel("Cos")
panels[2].set_ylabel("Cos*Exp")
panels[3].set_ylabel("Sin+Const")

# Set range
panels[-1].set_xlim(0,100)
panels[1].set_ylim(-5,5)

# Set log
panels[4].set_yscale("log")

# Plot
panels[0].plot(x, y_sin)
panels[1].plot(x, y_cos, linewidth=4)
panels[2].plot(x, y_cos_exp, color="red", linestyle="dashed",
               marker="o", markersize=5, markerfacecolor="orange")

# Plot with color-map colors
# autumn, bone, cool, copper, flag, gray, hot, hsv,
# jet, pink, prism, spring, summer, winter, spectral
for i in range(0,10):
	color=cm.winter(float(i) / 10)
	panels[3].plot(x,y_sin+i*100,color=color)

panels[4].hist(y_sin,bins=100,color="lightblue")

# Legend
panels[1].legend(["Cos"],loc="upper left")
panels[2].legend(["Sin*Exp"],loc="lower right")
plt.show()