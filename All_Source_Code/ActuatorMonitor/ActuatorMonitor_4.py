import tclab
import time
with tclab.TCLab() as lab:
    # simulated cyber attack
    print('Power Level (0-255): ' + str(lab.P1))

    # test cyber attack
    print('-'*40)
    print('Heater Power Off')
    print('Temperature (degC): ' + str(lab.T1))
    lab.P1=0      # set heater 1 power level to zero
    print('Power Level (0-255): ' + str(lab.P1))
    lab.Q1(100)   # turn on heater but no power (P1=0)
    print('Wait 30 sec')
    time.sleep(30)
    print('Temperature (degC): ' + str(lab.T1))

    print('-'*40)
    print('Heater Power On')
    print('Temperature (degC): ' + str(lab.T1))
    lab.P1 = 200  # restore heater 1 power level
    print('Power Level (0-255): ' + str(lab.P1))
    lab.Q1(100)   # turn on heater with power (P1=250)
    print('Wait 30 sec')
    time.sleep(30)
    print('Temperature (degC): ' + str(lab.T1))