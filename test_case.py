import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

P_cond = 4000   # example condenser pressure (Pa)

def equilibrium_uptake(T_bed):

    Ps_bed = PropsSI("P","T",T_bed,"Q",1,"Water")

    w_star = 0.346 * (P_cond/Ps_bed)**(1/1.6)

    return w_star

T_test = np.linspace(300, 420, 100)

w_test = []

for T in T_test:
    w_test.append(equilibrium_uptake(T))

plt.plot(T_test, w_test)

plt.xlabel("Temperature (K)")
plt.ylabel("w*")
plt.title("Equilibrium Uptake vs Temperature")

plt.grid()
plt.show()