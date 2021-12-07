import numpy as np
import pandas as pd
data = pd.read_csv('http://apmonitor.com/pds/uploads/Main/tclab_data6.txt')
data['log_time'] = np.log(data['Time'].values)
data.head()