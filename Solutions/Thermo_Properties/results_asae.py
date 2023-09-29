import matplotlib.pyplot as plt
import numpy as np

pos = ['TF (1000)','TF (300)','GEKKO','GEKKO (CPDL)']
train = [0.02877877,0.03747,0.03992,0.06099]
valid = [0.103778,0.0657985,0.06129,0.052896]
lg = ['TensorFlow (1000 epochs)','TensorFlow (300 epochs)','Python GEKKO','Constrained Predictive Deep Learning']

plt.subplot(2,1,1)
for i in range(4):
    if i<=1:
        plt.bar(x=pos[i],height=train[i],label=lg[i])
    else:
        plt.bar(x=pos[i],height=train[i],label=None)
plt.text(-0.4,0.033,'Over-fitting')
plt.text(2.6,0.065,'Worst training')
plt.legend(loc=2)
plt.ylim([0,0.08])
plt.ylabel(r'$\frac{1}{N} \sum{ \frac{\|y_p-y_m\|}{y_m} }$')

plt.subplot(2,1,2)
for i in range(4):
    if i>=2:
        plt.bar(x=pos[i],height=valid[i],label=lg[i])
    else:
        plt.bar(x=pos[i],height=valid[i],label=None)
plt.text(2.6,0.06,'Best validation')
plt.legend(loc=1)
plt.ylabel(r'$\frac{1}{N} \sum{  \frac{\|y_p-y_m\|}{y_m}}$')

plt.savefig('results_asae.eps')
plt.show()

##-------------- TensorFlow -------------------------------
##TensorFlow (1000 epochs - overtraining?)
##Mean sum abs diff - Training 0.028778776316279637
##Mean sum abs diff - Validate 0.10377812354052143
##
##TensorFlow (300 epochs)
##Mean sum abs diff - Training 0.03747426548575802
##Mean sum abs diff - Validate 0.06579851753218938
##
##--------------- GEKKO without constraint ----------------
##ms_abs train - 5 Neurons
##0.03992367902613089
##ms_abs validate
##0.061290126578834786
##
##--------------- GEKKO with constraint  w1 >= 0 ----------------
##ms_abs train
##0.0609934927226225
##ms_abs validate
##0.05289622331640575
