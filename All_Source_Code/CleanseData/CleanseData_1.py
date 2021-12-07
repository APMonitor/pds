import numpy as np
z = np.array([[      1,      2],
              [ np.nan,      3],
              [      4, np.nan],
              [      5,      6]])
iz = np.any(np.isnan(z), axis=1)
print(~iz)
z = z[~iz]
print(z)