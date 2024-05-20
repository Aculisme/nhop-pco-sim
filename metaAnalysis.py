import numpy as np
import pandas as pd

file_out = '/Users/lucamehl/Downloads/nhop-pco-sim/temp.txt'
df = pd.read_csv(file_out)

# 100_ms = df.where(df['time_until_synchronization'] == 100).dropna(

time_until_synch = df['time_until_synchronization_human_readable']#.values

print(time_until_synch)
time_until_synch_mean = np.mean(time_until_synch)
time_until_synch_st_dev = np.std(time_until_synch)
print("time_until_synch_mean", time_until_synch_mean)
print("time_until_synch_st_dev", time_until_synch_st_dev)
