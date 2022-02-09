from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('http://apmonitor.com/pds/uploads/Main/tclab_data6.txt')
data.set_index('Time',inplace=True)

# Split into train and test subsets (20% for test)
train, test = train_test_split(data, test_size=0.2, shuffle=False)