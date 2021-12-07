import pandas as pd
url = 'http://apmonitor.com/pds/uploads/Main/wind.txt'
data = pd.read_csv(url)
data.head()