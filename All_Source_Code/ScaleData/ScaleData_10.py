import pandas as pd
data = pd.read_csv('http://apmonitor.com/pds/uploads/Main/tclab_data6.txt')
data.set_index('Time',inplace=True)
data.plot(kind='hist',alpha=0.7,bins=30,figsize=(8,4))