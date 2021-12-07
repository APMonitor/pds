import pandas as pd
url = 'http://apmonitor.com/pds/uploads/Main/sonar_detection.txt'
data = pd.read_csv(url)

# One-hot enocde 'Class' (1 is 'Metal', 0 is 'Rock') with list comprehension
data.Class = [1 if x=='M' else 0 for x in data.Class]