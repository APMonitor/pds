import numpy as np
import pandas as pd

url = 'http://apmonitor.com/pds/uploads/Main/manufacturing.txt'
data = pd.read_csv(url)

# 'material' (1 is abs, 0 is pla) with numpy.where
data['material'] = np.where(data['material']=='abs',1,0)

# 'infill pattern' (1 is 'grid', 0 is 'honeycomb') with list comprehension
data.infill_pattern = [1 if ip=="grid" else 0 for ip in data.infill_pattern]