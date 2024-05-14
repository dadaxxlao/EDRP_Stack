import numpy as np
import matplotlib.pyplot as plt

# Constants
Q = 8e-9  # charge in coulombs (8 nC)
epsilon_0 = 8.854e-12  # permittivity of free space
R_0 = 1e-3  # radius in meters (1 mm)

# Function to calculate E_max
def E_max(D):
    D = D * 1e-3  # convert mm to meters
    term1 = Q / (4 * np.pi * epsilon_0 * R_0**2)
    term2 = Q / (4 * np.pi * epsilon_0 * (D - R_0)**2)
    return term1 + term2

# Values of D from 5 mm to 7 mm
D_values = np.linspace(5, 7, 300)  # 300 points between 5 mm and 7 mm

# Calculate E_max for each D
E_max_values = E_max(D_values)


# Recalculate E_max at specific points for annotations
D_specific = np.array([6, 7])  # specific values of D in mm
E_max_specific = E_max(D_specific)

# Re-plotting with annotations
plt.figure(figsize=(10, 5))
plt.plot(D_values, E_max_values, label='E_max(D)')
plt.scatter(D_specific, E_max_specific, color='red')  # mark specific points
for (d, e) in zip(D_specific, E_max_specific):
    plt.annotate(f'{e:.2e}', (d, e), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Distance D (mm)')
plt.ylabel('Electric Field E_max (N/C)')
plt.title('Theoretical value E_max vs Distance D')
plt.grid(True)
plt.legend()
plt.show()
