import numpy as np
import pandas as pd
tx = np.linspace(0,1,4); x  = np.cos(tx)
dx = pd.DataFrame({'Time':tx,'x':x})
tx = np.linspace(0,1,3)
x  = np.cos(tx)
dy = pd.DataFrame({'Time':tx,'x':x})