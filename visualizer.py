import numpy as np
import matplotlib.pyplot as plt

# Create a 15x15 matrix filled with zeros
matrix = np.zeros((15, 15))

# Define the regions with different colors
matrix[:3, :3] = 1  # Top 3x3 in green
matrix[6:9, 3:6] = 2  # Middle top 3x3 in blue
matrix[12:15, 12:15] = 2  # Bottom right 3x3 in blue
matrix[12:15, 6:9] = 2  # Bottom middle 3x3 in blue

# Plot the matrix
plt.imshow(matrix, cmap='viridis', interpolation='nearest')

# Set the tick labels for both x and y axes, ranging from 1 to 15
plt.xticks(np.arange(15), np.arange(1, 16))
plt.yticks(np.arange(15), np.arange(1, 16))

# Set the tick labels for the first 3 ticks to be '0'
plt.xticks(np.arange(0, 3), np.arange(0, 3))
plt.yticks(np.arange(0, 3), np.arange(0, 3))

# Adjust the tick parameters
plt.tick_params(axis='both', direction='out', which='both', length=6, width=2)

# Display the plot
plt.colorbar()
plt.show()


