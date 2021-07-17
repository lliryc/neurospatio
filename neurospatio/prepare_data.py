import pandas as pd
from matplotlib import pyplot
import numpy as np
sensor_data_df = pd.read_csv('../tests/data/CAF_VW_T_09012013.txt', sep ='\t')
sensors_df = pd.read_csv('../tests/data/CAF_Sensors.txt', sep ='\t')
sensors_df['EASTING'] = sensors_df['EASTING'].astype(int)
sensors_df['NORTHING'] = sensors_df['NORTHING'].astype(int)
minX, maxX = sensors_df['EASTING'].min(), sensors_df['EASTING'].max()
minY, maxY = sensors_df['NORTHING'].min(), sensors_df['NORTHING'].max()
sensors_df['EASTING'] = (sensors_df['EASTING'] - minX) // 2
sensors_df['NORTHING'] = (sensors_df['NORTHING'] - minY) // 2
width = (maxX - minX) // 2 + 1
height = (maxY - minY) // 2 + 1

grid = np.zeros((height, width))

for index, row in sensors_df.iterrows():
        grid[row['NORTHING']][row['EASTING']] = 1

pyplot.figure(figsize=(height,width))
pyplot.imshow(grid)
pyplot.colorbar()
pyplot.show()

