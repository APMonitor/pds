import pandas as pd
df = pd.read_csv('http://apmonitor.com/pds/uploads/Main/PV_BYU_South.txt')
factors=['Ambient Temperature (C)',
         'Wind Speed (m/s)', 
         'Plane of Array Irradiance (W/m^2)',
         'Cell Temperature (C)', 
         'DC Array Output (W)']