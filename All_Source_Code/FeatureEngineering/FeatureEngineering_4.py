import pandas as pd
url = 'http://apmonitor.com/pds/uploads/Main/animals.txt'
data = pd.read_csv(url)
data.head()