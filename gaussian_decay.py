import matplotlib.pyplot as plt
import numpy as np

# Gaussian Decay Function Logic
d0 = 20  # Travel time threshold
d = np.linspace(0, d0, 100)
weight = (np.exp(-0.5 * (d / d0)**2) - np.exp(-0.5)) / (1 - np.exp(-0.5))

plt.figure(figsize=(8, 5))
plt.plot(d, weight, color='#007BFF', lw=3)
plt.title('Gaussian Distance Decay ($d_0 = 20$ min)', fontsize=14)
plt.xlabel('Travel Time (min)', fontsize=12)
plt.ylabel('Accessibility Weight ($G$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show() # Run this and screenshot it!