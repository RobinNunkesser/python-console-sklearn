import numpy as np
import matplotlib.pyplot as plt

def ground_truth_rs(x1, x2):
    return (0.8 * (x1 < 6) * (x2 > 6) + 0.3 * (x1 < 5) * (x2 < 7) + 0.8 * (x1 > 4) * (x2 < 4)) / 1.1

def ground_truth_rl(x1, x2):
    result = np.zeros_like(x1, dtype=float)
    result[x1 > -0.1] = 0.5    
    result[x1 < 5] = 0.3
    result[(x1 >= 5) & (x2 < 4)] = 0.9
    result[(x1 >= 5) & (x2 >= 4) & (x1 > 6)] = 0.7
    result[(x1 >= 5) & (x2 >= 4) & (x1 <= 6)] = 0.5    
    return result
    
def ground_truth_rt(x1, x2):
    result = np.zeros_like(x1, dtype=float)
    result[x1 > 5] = 0.8
    result[(x1 > 5) & (x2 > 4)] = 0.6
    result[x1 <= 5] = 0.2
    result[(x1 <= 5) & (x2 > 6)] = 0.9
    return result
    

# Create a grid of x1 and x2 values
x1 = np.linspace(0, 10, 1000)
x2 = np.linspace(0, 10, 1000)
X1, X2 = np.meshgrid(x1, x2)

# Compute the ground truth probabilities

Z = {
    "set":  ground_truth_rs(X1, X2),
    "list": ground_truth_rl(X1, X2),
    "tree": ground_truth_rt(X1, X2),
}

# Create the heatmap with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (key, ax) in enumerate(zip(["set", "list", "tree"], axes)):
    contour = ax.contourf(X1, X2, Z[key], levels=np.linspace(0, 1, 20))
    fig.colorbar(contour, ax=ax, label='Score', ticks=[])    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Rule ({key})')

plt.tight_layout()
plt.show()

