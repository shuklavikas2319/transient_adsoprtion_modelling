import numpy as np
from scipy.integrate import solve_ivp
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\shukl\OneDrive\Desktop\python\adsorption\thermodynamics')
from state_point_calculator import rootfind
import desorption as des
obj=rootfind()
R_vapor = 461.5 # J/(kg*K), gas constant for DA
T_init = des.ambient_temperature(des.t_switchdes)
m_b = des.m_b
m_a = des.m_a

# Parameters
H_ads= 2.8e6  # J/kg, heat of adsorption
Cp_b = 460 # specific heat of adsorber bed metal (J/kg-K)
Cp_a = 920 # specific heat of adsorbent (J/kg-K)
Cp_l = 4180 # specific heat of liquid water (J/kg-K)
D_so = 2.54e-4 # m^2/s, diffusivity of water in the adsorbent
E_a = 42000 # J/MOL, activation energy 
R= 8.314 # J/(mol*K), universal gas constant
R_p = 1.7e-4 # m, radius of pore of adsorbent 
F_o = 15 # fourier number
alpha = 0.9 # adsorptivity coefficient
A = (50*58/10000) # m^2, surface area of glass cover (50cm x 58cm)
U = 4 # W/m2K, overall heat transfer coefficient between adsorber bed and ambient
l = 58/100 # m, length of the adsorber bed
b = 50/100 # M , breadth of the adsorber bed
h = 60/100 # m , height of the adsorber bed
A_surf = 2*(l*b + l*h + b*h) # m^2, surface area of the adsorber bed
T0 = des.T_switchdes   # Initial temperature in Kelvin 
w0 = des.w_switchdes     # Initial uptake (kg/kg)
T_night = 30 + 273.15  # choose consistent Kelvin
tau = 5.5 * 3600
def ambient_temperature_night(t):
    return T_night + (T_init - T_night) * np.exp(-t / tau) # Ambient temperature varies between 298 K (25°C) and 318 K (45°C) over the course of the day, peaking at around 2 PM (t=14)
def P_b(T, w):
    Psat = PropsSI("P","T",T,"Q",1,"Water")
    return Psat*(w/0.346)**1.6
def eqns(t, vars):
    T = vars[0]
    dT_dt = -U*A_surf*(T-ambient_temperature_night(t))/(m_b*Cp_b + m_a*(Cp_a + w0*Cp_l)) 
    return [dT_dt]
# Initial conditions
T0 = des.T_switchdes   # Initial temperature in Kelvin 
P_evap = PropsSI("P","T",obj.T_evap,"Q",1,"Water")

def event_evaporator_connection(t, y):

    T = y[0]

    return P_b(T, w0) - P_evap

event_evaporator_connection.terminal = True
event_evaporator_connection.direction = -1
t_span = (des.t_switchdes, 24*3600)

solution = solve_ivp(eqns,t_span,[T0],method='BDF',events=event_evaporator_connection,rtol=1e-6,atol=1e-9)
T_end = solution.y[0,-1]

if len(solution.t_events[0]) == 0:
    print("No switching event detected!")
else:
    T_switchdes = solution.y_events[0][0][0] # Temperature at which switching occurs
    t_switchdes = solution.t_events[0][0]    # Time at which switching occurs
    
    T_end = solution.y[0,-1]
    if __name__ == "__main__":
        print(f"Adsorption starts after {t_switchdes/3600:.2f} h")
        print(f"Bed temperature = {T_switchdes-273.15:.2f} °C")

        

        plt.plot(solution.t/3600, solution.y[0]-273.15)
        plt.xlabel("Time (h)")
        plt.ylabel("Bed Temperature (°C)")
        plt.grid()
        plt.show()
