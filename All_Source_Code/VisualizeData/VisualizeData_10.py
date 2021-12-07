import matplotlib.pyplot as plt
plt.scatter(data['Q1'],data['T1'])

# add labels and title
plt.xlabel('Heater (%)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()