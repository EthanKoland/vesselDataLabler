import numpy as np
import matplotlib

t0 = 1
t1 = 10
n = 5
lin = np.linspace(t0, t1, n, dtype=int)
lin2 = np.linspace(0, 1, n)
cmap = matplotlib.cm.get_cmap('nipy_spectral')
print(cmap(lin2))
# lin2 = np.linspace(0, 1, n)
print(np.linspace(t0, t1, n, dtype=int))