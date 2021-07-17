"""
Code tests
"""
import sys
import os
import numpy as np
import pandas as pd
from neurospatio.learner2D import SpLearner, SpreadOp
from matplotlib import pyplot

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def data():

    vw_t_file = os.path.join(BASE_DIR, "data", "CAF_VW_T_09012013.txt")
    vw_t_df = pd.read_csv(vw_t_file, sep='\t')

    sensors_file = os.path.join(BASE_DIR, "data", "CAF_Sensors.txt")
    sensors_df = pd.read_csv(sensors_file, sep='\t')
    res_df = pd.concat([vw_t_df, sensors_df], axis=1, join="inner")
    train_df = res_df[["EASTING", "NORTHING", "T_30cm", "VW_30cm"]]
    train_df["EASTING"] = train_df["EASTING"].astype(int)
    train_df["NORTHING"] = train_df["NORTHING"].astype(int)
    train_df["T_30cm"] = train_df["T_30cm"].astype(float)
    train_df["VW_30cm"] = train_df["VW_30cm"].astype(float)
    min_x, min_y = train_df["EASTING"].min() - 50, train_df["NORTHING"].min() - 50
    train_df["EASTING"] = (train_df["EASTING"] - min_x) // 5
    train_df["NORTHING"] = (train_df["NORTHING"] - min_y) // 5

    return train_df

def aux_grid_data():
    aux_df = pd.read_csv("data/CAF_VW_Interpolated.txt", sep=" ", names=["EASTING", "NORTHING", "VW_30cm"])
    return aux_df

import numpy as np
from neurospatio.learner2D import SpLearner, SpreadOp

def example1():
    # train dataset: set of 2D points with values
    points = np.array([
        [2, 8], # X, Y coordinate
        [8, 10],
        [10,2]
    ])
    values = np.array([19.39, 17.18, 20.95])
    # create SpLearner with radial interpolation
    learner = SpLearner(points, values, spread_op_flag=SpreadOp.CENTROIDS, n_epochs=600)
    # generate a grid for the test
    grid = np.array([[i, j] for i in range(10) for j in range(10)])
    # [
    # [0,0], ...
    #  [9,9]
    #]
    predicted_values = learner.execute(grid_points=grid)
    # Example of prediction (!may be not the same)
    #[
    # [17.96],
    # ...
    # [16.27]
    # ]

def base_test():
    train_df = data()

    points = train_df[["EASTING", "NORTHING"]].values
    values = train_df["T_30cm"].values

    learner = SpLearner(points, values, spread_op_flag=SpreadOp.CENTROIDS)

    x = train_df["EASTING"].values
    y = train_df["NORTHING"].values

    width = max(x) + 1
    height = max(y) + 1

    grid_points = np.array([[i,j] for i in range(width + 10) for j in range(height + 10)])
    grid_vals = learner.execute(grid_points)

    grid_df = pd.DataFrame(data = grid_points, columns=["X", "Y"], dtype=int)
    grid_df["val"] = grid_vals

    grid_vals = pd.pivot_table(grid_df, values="val", index=["Y"], columns=["X"]).values

    pyplot.figure(figsize=(height + 10, width + 10))
    im = pyplot.imshow(grid_vals, vmin=min(values), vmax=max(values))
    pyplot.scatter(x, y, s=100, c='black', alpha=None, marker='.')
    train_df_values = train_df.values
    for r in train_df_values:
        pyplot.annotate(r[2], (r[0], r[1]))
    pyplot.colorbar(im)
    pyplot.show()

def influence_test():
    train_df = data()

    points = train_df[["EASTING", "NORTHING"]].values
    values = train_df["T_30cm"].values

    learner = SpLearner(points, values, influence_max=1000)

    x = train_df["EASTING"].values
    y = train_df["NORTHING"].values

    width = max(x) + 50
    height = max(y) + 50

    grid_points = np.array([[i,j] for i in range(width) for j in range(height)])
    grid_vals = learner.execute(grid_points)

    grid_df = pd.DataFrame(data = grid_points, columns=["X", "Y"], dtype=int)
    grid_df["val"] = grid_vals

    grid_vals = pd.pivot_table(grid_df, values="val", index=["Y"], columns=["X"]).values

    pyplot.figure(figsize=(height, width))
    im = pyplot.imshow(grid_vals, vmin=min(values), vmax=max(values))
    pyplot.scatter(x, y, s=100, c='black', alpha=None, marker='.')
    train_df_values = train_df.values
    for r in train_df_values:
        pyplot.annotate(r[2], (r[0], r[1]))
    pyplot.colorbar(im)
    pyplot.show()

def base_test2():
    train_df = data()[["EASTING", "NORTHING","VW_30cm"]]

    points = train_df[["EASTING", "NORTHING"]].values
    values = train_df["VW_30cm"].values

    learner = SpLearner(points, values * 100, spread_op_flag = SpreadOp.CENTROIDS, n_epochs=20000) # , spread_op_flag = SpreadOp.CENTROIDS

    x = train_df["EASTING"].values
    y = train_df["NORTHING"].values

    width = max(x) + 50
    height = max(y) + 50

    grid_points = np.array([[i,j] for i in range(width) for j in range(height)])
    grid_vals = learner.execute(grid_points)
    grid_vals = grid_vals / 100.0

    grid_df = pd.DataFrame(data = grid_points, columns=["X", "Y"], dtype=int)
    grid_df["val"] = grid_vals

    grid_vals = pd.pivot_table(grid_df, values="val", index=["Y"], columns=["X"]).values

    pyplot.figure(figsize=(height, width))
    im = pyplot.imshow(grid_vals, vmin=min(values), vmax=max(values))
    pyplot.scatter(x, y, s=100, c='black', alpha=None, marker='.')
    train_df_values = train_df.values
    for r in train_df_values:
        pyplot.annotate(f"  {r[2]}", (r[0], r[1]))
    pyplot.colorbar(im)
    pyplot.show()
    # export to file
    grid_df.to_csv("data/CAF_VW_Interpolated.txt", sep=' ', header=False, index=False)

def aux_test():
    train_df = data()

    points = train_df[["EASTING", "NORTHING"]].values
    values = train_df["T_30cm"].values
    aux_values = train_df[["VW_30cm"]].values

    learner = SpLearner(points, values, aux_values*100, spread_op_flag = SpreadOp.CENTROIDS, n_epochs=10000) # , spread_op_flag = SpreadOp.CENTROIDS

    x = train_df["EASTING"].values
    y = train_df["NORTHING"].values

    width = max(x) + 50
    height = max(y) + 50

    grid_points = np.array([[i,j] for i in range(width) for j in range(height)])

    aux_grid_vals = aux_grid_data()[["VW_30cm"]]

    grid_vals = learner.execute(grid_points, aux_grid_vals*100)

    grid_df = pd.DataFrame(data = grid_points, columns=["X", "Y"], dtype=int)
    grid_df["val"] = grid_vals

    grid_vals = pd.pivot_table(grid_df, values="val", index=["Y"], columns=["X"]).values

    pyplot.figure(figsize=(height, width))
    im = pyplot.imshow(grid_vals, vmin=min(values), vmax=max(values))
    pyplot.scatter(x, y, s=100, c='black', alpha=None, marker='.')
    train_df_values = train_df[["EASTING", "NORTHING", "T_30cm"]].values
    for r in train_df_values:
        pyplot.annotate(f"  {r[2]}", (r[0], r[1]))
    pyplot.colorbar(im)
    pyplot.show()
    # export to file
    grid_df.to_csv("data/CAF_T_Interpolated.txt", sep=' ', header=False, index=False)

example1()
#aux_test()
#base_test2()
#influence_test()
pass



