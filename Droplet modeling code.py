import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
gamma = 72e-3
g = 9.807
rho = 997
c = rho * g / gamma
V_e = 0.0892e-6
L = 1e-3
ds = L * 0.001
epsilon = 1e-10

# Differential equations
def equations(s, y, r, c, epsilon):
    x, z, theta, V = y
    dxds = np.cos(theta)
    dzds = np.sin(theta)
    dthetads = 2 / r + c * z - np.sin(theta + epsilon) / (x + r * epsilon)
    dVds = np.pi * x**2 * np.sin(theta)
    return [dxds, dzds, dthetads, dVds]

# Initial conditions
y0 = [0, 0, 0, 0]

# Solver function
def solve_ode(r):
    sol = solve_ivp(equations, [0, 10 * L], y0, args=(r, c, epsilon), dense_output=True)
    return sol

# Theta_e values in degrees
theta_e_values = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

# Initial values of s0 and r0 for each theta_e calculation
s0_initial_values = [5 * L, 5 * L, 5 * L, 5 * L, 5 * L, 5 * L, 4 * L, 5 * L, 4 * L, 4 * L, 4 * L]
r0_initial_values = [9 * L, 8.5 * L, 8 * L, 7 * L, 6 * L, 6 * L, 6 * L, 5 * L, 4 * L, 3 * L, 2 * L]

# Store data for plotting
curves = []
s0_evolution = []
r0_evolution = []

# Iterate over different theta_e values
for theta_e_deg, s0_init, r0_init in zip(theta_e_values, s0_initial_values, r0_initial_values):
    theta_e = theta_e_deg * np.pi / 180
    s0 = s0_init
    r0 = r0_init

    # Store solutions for comparison
    s0_values = []
    r0_values = []

    # Newton-Raphson iteration
    for i in range(10):
        sol_r0 = solve_ode(r0)
        V_s0_ds = sol_r0.sol(s0 + ds)[3]
        V_s0 = sol_r0.sol(s0)[3]
        theta_s0_ds = sol_r0.sol(s0 + ds)[2]
        theta_s0 = sol_r0.sol(s0)[2]

        Fd11 = (V_s0_ds - V_s0) / ds
        Fd12 = (solve_ode(r0 + ds).sol(s0)[3] - V_s0) / ds
        Fd21 = (theta_s0_ds - theta_s0) / ds
        Fd22 = (solve_ode(r0 + ds).sol(s0)[2] - theta_s0) / ds

        Fd = np.array([[Fd11, Fd12], [Fd21, Fd22]])

        F1 = V_s0 - V_e
        F2 = theta_s0 - theta_e

        F = np.array([F1, F2])

        delta = np.linalg.solve(Fd, F)
        s0, r0 = s0 - delta[0], r0 - delta[1]

        s0_values.append(s0)
        r0_values.append(r0)

        print(f"theta_e={theta_e_deg}, i={i}, s={s0}, r={r0}")

    # Results
    sol_final = solve_ode(r0)
    x_s0, z_s0, theta_s0_final, V_s0_final = sol_final.sol(s0)
    theta_s0_final_deg = np.degrees(theta_s0_final)

    print(f"Theta_e={theta_e_deg}: x_s0={x_s0}, z_s0={z_s0}, theta_s0_final_deg={theta_s0_final_deg}, V_s0_final={V_s0_final}")

    # Collect curve data
    s_vals = np.linspace(0, s0, 1000)
    x_vals, z_vals = sol_final.sol(s_vals)[:2]
    x_vals_neg = -x_vals
    x_combined = np.concatenate((x_vals_neg[::-1], x_vals))
    z_combined = np.concatenate((z_vals[::-1], z_vals))

    # Adjust the vertical position of each curve
    z_offset = z_combined[-1]  # Get the last z value to adjust
    z_combined_adjusted = z_combined - z_offset  # Adjust the entire curve

    curves.append((x_combined, z_combined_adjusted, theta_e_deg))

    # Collect evolution data
    s0_evolution.append(s0_values)
    r0_evolution.append(r0_values)

# Plot all combined curves
plt.figure(figsize=(14, 8))
for x_combined, z_combined, theta_e_deg in curves:
    plt.plot(x_combined, -z_combined, label=f'Theta_e = {theta_e_deg:.2f} degrees')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Combined Curves for Different Theta_e Values')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio
plt.show()

# Plot evolution of s0 and r0
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
for s0_values, theta_e_deg in zip(s0_evolution, theta_e_values):
    plt.plot(range(10), s0_values, label=f'Theta_e = {theta_e_deg:.2f} degrees')
plt.xlabel('Iteration')
plt.ylabel('s0')
plt.title('Evolution of s0')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for r0_values, theta_e_deg in zip(r0_evolution, theta_e_values):
    plt.plot(range(10), r0_values, label=f'Theta_e = {theta_e_deg:.2f} degrees')
plt.xlabel('Iteration')
plt.ylabel('r0')
plt.title('Evolution of r0')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

