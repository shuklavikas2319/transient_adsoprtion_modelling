import numpy as np
from scipy.integrate import solve_ivp
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\shukl\OneDrive\Desktop\python\adsorption\thermodynamics')
from state_point_calculator import rootfind
obj=rootfind()
R_vapor = 461.5 # J/(kg*K), gas constant for DA
def solar_irradiance(t):
   #maximum solar irradiance at noon (t=12) is approximately 270.8 W/m^2, but we can use a simplified model
    return max(0, 900 * np.sin(np.pi * t / (12*3600)))  # Peak at noon (t=12) in watts per square meter
def equilibrium_uptake(T_bed):

    Ps_cond = PropsSI("P","T",obj.T_cond,"Q",1,"Water")
    Ps_bed  = PropsSI("P","T",T_bed,"Q",1,"Water")
    w_0 = 0.3413 # kg/kg, maximum uptake at saturation
    w_star = 0.346 * (Ps_cond/Ps_bed)**(1/1.6)
    w_star = min(w_star, w_0)

    return w_star

def ambient_temperature(t):

   return 308 + 10*np.sin(np.pi*(t)/(12*3600))  # Ambient temperature varies between 298 K (25°C) and 318 K (45°C) over the course of the day, peaking at around 2 PM (t=14)

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
T_ambient = 318.15 # K, ambient temperature
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
    dT_dt = (m_a*H_ads*dw_dt + solar_irradiance(t)*alpha*A-U*A_surf*(T-ambient_temperature(t)))/(m_b*Cp_b + m_a*(Cp_a + w*Cp_l)) 
    return [dT_dt, dw_dt]
# Initial conditions
T0 = 278.15  # Initial temperature in Kelvin (20°C)
w0 = 0.3413     # Initial uptake (kg/kg)
initial_conditions = [T0, w0]
# Time span for the simulation (0 to 12 hours)
t_span = (0, 12*3600)  # seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)
# Solve the system of ODEs
solution = solve_ivp(eqns, t_span, initial_conditions, t_eval=t_eval , method='Radau',rtol=1e-6, atol=1e-9)

# Extract results
T_solution = solution.y[0]  # Temperature over time
w_solution = solution.y[1]  # Uptake over time
time_hours = solution.t / 3600
w_star_solution = []

for T in T_solution:
    w_star_solution.append(equilibrium_uptake(T))

T_max = np.max(T_solution)
T_max_index = np.argmax(T_solution)

time_T_max = time_hours[T_max_index]



w_min = np.min(w_solution)
w_min_index = np.argmin(w_solution)
time_w_min = time_hours[w_min_index]
T# Export switching state variables
T_switch = T_solution[w_min_index]
w_switch = w_solution[w_min_index]
t_switch = solution.t[w_min_index]
if __name__ == "__main__":
    print(f"Maximum bed temperature = {T_max - 273.15:.2f} °C")
    print(f"Occurs at t = {time_T_max:.2f} hours")
    print(f"Minimum uptake = {w_min:.4f} kg/kg")
    print(f"Occurs at t = {time_w_min:.2f} hours")
    print(f"Corresponding bed temperature at minimum uptake = {T_switch - 273.15:.2f} °C")
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