import numpy as np
from scipy.integrate import solve_ivp
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\shukl\OneDrive\Desktop\python\adsorption\thermodynamics')
from state_point_calculator import rootfind
import isosteric_cooling as iso
import desorption as des
obj=rootfind()
R_vapor = 461.5 # J/(kg*K), gas constant for DA

def equilibrium_uptake(T_bed):

    Ps_evap = PropsSI("P","T",obj.T_evap,"Q",1,"Water")
    Ps_bed  = PropsSI("P","T",T_bed,"Q",1,"Water")
    w_star = 0.346 * (Ps_evap/Ps_bed)**(1/1.6)

    return w_star
T_init = iso.ambient_temperature_night(iso.t_switchdes)
T_night = 30 + 273.15  # choose consistent Kelvin
tau = 5.5 * 3600
def ambient_temperature_nightp(t):
    return T_night + (T_init - T_night) * np.exp(-t / tau) # Ambient temperature varies between 298 K (25°C) and 318 K (45°C) over the course of the day, peaking at around 2 PM (t=14)

m_b = iso.m_b
m_a = iso.m_a

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

# define system of ODEs
def eqns(t, vars):
    T = vars[0]
    w = vars[1]
    w_star = equilibrium_uptake(T)
    dw_dt = (F_o*D_so/(R_p**2)*np.exp(-E_a/(R*T))*(w_star - w))
    dT_dt = (m_a*H_ads*dw_dt -U*A_surf*(T-ambient_temperature_nightp(t)))/(m_b*Cp_b + m_a*(Cp_a + w*Cp_l)) 
    return [dT_dt, dw_dt]
# Initial conditions
T0 = iso.T_switchdes   # Initial temperature in Kelvin 
w0 = des.w_switchdes     # Initial uptake (kg/kg)
initial_conditions = [T0, w0]
# Time span for the simulation (0 to 12 hours)
t_span = (iso.t_switchdes, 24*3600)  # seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)
# Solve the system of ODEs
solution = solve_ivp(eqns, t_span, initial_conditions, t_eval=t_eval , method='BDF',rtol=1e-6, atol=1e-9)

# Extract results
T_solution = solution.y[0]  # Temperature over time
w_solution = solution.y[1]  # Uptake over time
time_hours = solution.t / 3600
w_star_solution = []

for T in T_solution:
    w_star_solution.append(equilibrium_uptake(T))

T_min = np.min(T_solution)
T_min_index = np.argmin(T_solution)

time_T_min = time_hours[T_min_index]



w_max = np.max(w_solution)
w_max_index = np.argmax(w_solution)
time_w_max = time_hours[w_max_index]

print(f"Minimum bed temperature = {T_min - 273.15:.2f} °C")
print(f"Occurs at t = {time_T_min:.2f} hours")

print(f"Maximump uptake = {w_max:.4f} kg/kg")
print(f"Occurs at t = {time_w_max:.2f} hours")
# Plotting results
plt.figure(figsize=(12, 6))

# Temperature subplot
plt.subplot(2, 1, 1)
plt.plot(solution.t / 3600, T_solution - 273.15, label='Temperature')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature of Adsorber Bed Over Time')
plt.legend()

# Uptake subplot
plt.subplot(2, 1, 2)
plt.plot(solution.t / 3600, w_solution, label='Actual uptake')
plt.plot(solution.t / 3600, w_star_solution, label='Equilibrium uptake')
plt.xlabel('Time (hours)')
plt.ylabel('Uptake (kg/kg)')
plt.title('Uptake of Adsorbent Over Time')
plt.legend()

plt.tight_layout()
plt.show()