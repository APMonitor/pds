import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import plotly.express as px

df = pd.read_csv('http://apmonitor.com/pds/uploads/Main/PV_BYU_South.txt')
factors=['Ambient Temperature (C)',
         'Wind Speed (m/s)', 
         'Plane of Array Irradiance (W/m^2)',
         'Cell Temperature (C)', 
         'DC Array Output (W)']
print(df.columns)
data = df[factors].copy() # take only subset of data columns
# remove rows where there is no sunlight
data = data[data['Plane of Array Irradiance (W/m^2)']>0.01]
# calculate efficiency (use PV Cell m^2 to get true efficiency)
data['efficiency'] = data['DC Array Output (W)'] \
                      /data['Plane of Array Irradiance (W/m^2)']
print(data.head())
print(data.describe())
data.plot()

profile = ProfileReport(data, explorative=True, minimal=False)
try:
   profile.to_widgets()            # view as widget in Notebook
except:
   profile.to_file('PV_Data.html') # html file if widget not available

fig = px.scatter(data, x="Ambient Temperature (C)", \
                       y="Cell Temperature (C)")
fig.show()
sns.pairplot(data)

plt.figure()
x = data['Ambient Temperature (C)']
y = data['Cell Temperature (C)']
plt.scatter(x,y)
plt.xlabel('Ambient Temperature (°C)')
plt.ylabel('Cell Temperature (°C)')
plt.show()