import numpy as np
import matplotlib.pyplot as plt

def ground_truth_p(x1, x2):
    return 0.8 * (x1 < 6) * (x2 > 6) + 0.3 * (x1 < 5) * (x2 < 7) + 0.8 * (x1 > 4) * (x2 < 4)

# Create a grid of x1 and x2 values
x1 = np.linspace(0, 10, 1000)
x2 = np.linspace(0, 10, 1000)
X1, X2 = np.meshgrid(x1, x2)

# Compute the ground truth probabilities
Z = ground_truth_p(X1, X2)

# Create the heatmap
plt.figure(figsize=(8, 6))
#plt.contourf(X1, X2, Z, levels=20, cmap='RdYlBu_r')
plt.contourf(X1, X2, Z, levels=20)
plt.colorbar(label='Score')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Ground Truth Probability')
plt.show()