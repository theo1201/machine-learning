import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
x = np.linspace(0,2*np.pi,400)
y =np.sin(x**2)

# fig,ax = plt.subplots()
# ax.plot(x,y)
# ax.set_title('Simple plot')
# plt.show()
#
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y)
# plt.show()
# fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
# axes[0, 0].plot(x, y)
# axes[1, 1].scatter(x, y)

# plt.subplots(2, 2, sharex='col')
# plt.subplots(2, 2, sharey='row')
# plt.subplots(2, 2, sharex='all', sharey='all')

# plt.subplots(2, 2, sharex=True, sharey=True)

x = np.arange(1, 7).reshape(2, 3)
print(x.flat[3])