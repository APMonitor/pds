import numpy as np
import pandas as pd

tx = np.linspace(0,1,4); x  = np.cos(tx)
dx = pd.DataFrame({'Time':tx,'x':x})

ty = np.linspace(0,1,3); y  = np.sin(ty)
dy = pd.DataFrame({'Time':ty,'y':y})