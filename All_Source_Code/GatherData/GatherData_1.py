import numpy as np
import pandas as pd
tx = np.linspace(0,1,8); x  = np.cos(tx)
dx = pd.DataFrame({'Time':tx,'x':x})
dx.to_csv('dx.csv',index=False)
print(dx)