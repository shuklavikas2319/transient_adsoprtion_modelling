import numpy as np
from scipy.integrate import solve_ivp
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\shukl\OneDrive\Desktop\python\adsorption\thermodynamics')
from state_point_calculator import rootfind
obj=rootfind()
def equilibrium_uptake(T_bed):

    Ps_evap = PropsSI("P","T",obj.T_evap,"Q",1,"Water")
    Ps_bed  = PropsSI("P","T",T_bed,"Q",1,"Water")
    w_0 = 0.3413 # kg/kg, maximum uptake at saturation
    w_star = 0.346 * (Ps_evap/Ps_bed)**(1/1.6)

    return w_star
def solar_irradiance(t):
   #maximum solar irradiance at noon (t=12) is approximately 270.8 W/m^2, but we can use a simplified model
    return max(0, 900 * np.sin(np.pi * t / (12*3600)))  # Peak at noon (t=12) in watts per square meter
R_vapor = 461.5 # J/(kg*K), gas constant for DA
T_init = 308.15  # Initial temperature in Kelvin (35°C)
m_b = float(input("Enter the mass of adsorber bed m_b: "))
m_a = float(input("Enter the mass of adsorbent m_a: "))


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
T0 = T_init   # Initial temperature in Kelvin 
w0 = 0.3413 # Initial uptake (kg/kg)
def ambient_temperature(t):

   return 308 + 10*np.sin(np.pi*(t)/(12*3600))  # Ambient temperature varies between 298 K (25°C) and 318 K (45°C) over the course of the day, peaking at around 2 PM (t=14)
def P_b(T, w):
    Psat = PropsSI("P","T",T,"Q",1,"Water")
    return Psat*(w/0.346)**1.6
def eqns(t, vars):
    T = vars[0]
    dT_dt =  (solar_irradiance(t)*alpha*A-U*A_surf*(T-ambient_temperature(t)))/(m_b*Cp_b + m_a*(Cp_a + w0*Cp_l)) 
    return [dT_dt]
# Initial conditions
T0 = T_init   # Initial temperature in Kelvin 
P_cond = PropsSI("P","T",obj.T_cond,"Q",1,"Water")

def event_evaporator_connection(t, y):

    T = y[0]

    return P_cond - P_b(T, w0)

event_evaporator_connection.terminal = True
event_evaporator_connection.direction = -1
t_span = (0, 24*3600)
# solving ivp with event detection for switching

solution = solve_ivp(
    eqns,
    t_span,
    [T0],
    method='BDF',
    events=event_evaporator_connection,
    rtol=1e-6,
    atol=1e-9
)
if len(solution.t_events[0]) == 0:
    print("No switching event detected!")
    T_ads_start = None
    t_ads_start = None
else:
    T_ads_start = solution.y_events[0][0][0] # Temperature at which switching occurs
    t_ads_start = solution.t_events[0][0]    # Time at which switching occurs

if __name__ == "__main__":
    print(f"Switch temperature = {T_ads_start-273.15:.2f} °C")
    print(f"Switch time = {t_ads_start/3600:.2f} h")
    
        
    plt.plot(solution.t/3600, solution.y[0]-273.15)
    plt.xlabel("Time (h)")
    plt.ylabel("Bed Temperature (°C)")
    plt.grid()
    plt.show()
