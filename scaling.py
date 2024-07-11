import matplotlib.pyplot as plt
import pandas as pd

# Create DataFrame from the provided data
data = {
    "Number of nodes": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    "Number of broadcasts": [26, 30, 48, 66, 78, 92, 116, 130, 154, 172, 188, 220, 242, 264, 264, 286, 292, 316, 344, 360, 374, 408, 424, 442, 462, 490, 520, 544, 568, 592, 602],
    "Time to synchronize (ms)": [85.2, 185.6, 268.3, 315.6, 319.8, 443.7, 504.2, 528.9, 548.8, 577.2, 656.5, 753.1, 838.1, 872.3, 898.6, 940, 1039.4, 1026.3, 1125.5, 1161.2, 1225.5, 1223.7, 1297.2, 1378.1, 1458.1, 1493.9, 1649.4, 1649.4, 1680.3, 1649.4, 1694.7]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

# Plot for Number of broadcasts
plt.plot(df["Number of nodes"], df["Number of broadcasts"], marker='o', color='b', label='Number of broadcasts')

# Plot for Time to synchronize (ms)
plt.plot(df["Number of nodes"], df["Time to synchronize (ms)"], marker='x', color='r', label='Time to synchronize (ms)')
plt.legend(loc='upper left')
# Titles and labels
plt.title('SyncWave Prototype: Scaling with Number of Nodes in a Path Graph')
plt.xlabel('Number of Nodes')
plt.ylabel('Number of broadcasts')
plt.grid(True)

# Secondary y-axis for the Time to synchronize (ms)
plt.twinx()
# plt.plot(df["Number of nodes"], df["Time to synchronize (ms)"], marker='x', color='r', linestyle='--')
plt.ylabel('Time to synchronize (ms)')


# Adding legends
# plt.legend()

# Adding a grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.show()
