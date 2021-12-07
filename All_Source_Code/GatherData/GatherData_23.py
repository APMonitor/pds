import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# stock ticker symbol
url = 'http://apmonitor.com/pds/uploads/Main/goog.txt'

# import data with pandas
data = pd.read_csv(url)
print(data.describe())

# calculate change and volatility
data['Change'] = data['Close']-data['Open']
data['Volatility'] = data['High']-data['Low']
analysis = ['Open','Volume','Volatility','Change']
sns.heatmap(data[analysis].corr())
plt.show()