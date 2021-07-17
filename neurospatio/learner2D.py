from enum import IntFlag
import numpy as np
import tensorflow as tf
import random as rn
rn.seed(21)
np.random.seed(42)
tf.random.set_seed(24)
import scipy.spatial as spatial
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from .get_best_callback import GetBest

class SpreadOp(IntFlag):
    NONE = 0,
    HORIZONTAL = 1,
    VERTICAL = 2,
    CENTROIDS = 4,
    NEIGHBORS = 8

class SpLearner:

    # SpreadOp.HORIZONTAL | SpreadOp.VERTICAL |
    # SpreadOp.CENTROIDS
    # | SpreadOp.NEIGHBORS
    # | SpreadOp.HORIZONTAL
    def __init__(self, points, values, auxiliary_values = None, spread_op_flag = (SpreadOp.HORIZONTAL | SpreadOp.VERTICAL | SpreadOp.CENTROIDS) , \
                 influence_max = None, fcl0 = 300, fcl1 = 150, fcl2 = 75, droupout = 0.5, n_epochs = 6000, batch_size = 128,  \
                 regularizer_l1 = 1e-4, regularizer_l2 = 1e-3, train_eval_split_random_state = 1, eval_ratio = None,  \
                 verbose_mode = 1, inference_workers = 10):
        self.spread_op_flag = spread_op_flag
        self.influence_max = influence_max
        self.fcl0 = fcl0
        self.fcl1 = fcl1
        self.fcl2 = fcl2
        self.dropout = droupout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.regularizer_l1 = regularizer_l1
        self.regularizer_l2 = regularizer_l2
        self.centroids = points
        self.values = values
        self.auxiliary_values = auxiliary_values
        self.train_eval_split_random_state = train_eval_split_random_state
        self.eval_ratio = eval_ratio
        self.verbose_mode = verbose_mode
        self.inference_workers = inference_workers

    def _align(self, list, max_len):
        lst_len = len(list)
        np.argsort(list)

    def _invert(self, arr):
        mx = np.max(arr)
        mn = np.min(arr)
        return -(arr - mn) + mx

    def _transform(self, X, test_start_ix, C=None):
        Xt = X.transpose()
        X_c = None
        nb_ixs = None

        features = []
        test_data_influence_free_ixs = []

        if self.spread_op_flag & SpreadOp.HORIZONTAL == SpreadOp.HORIZONTAL:
            features.append(np.array([Xt[0]]))
            features.append(np.array([Xt[0], self._invert(Xt[0])]))

        if self.spread_op_flag & SpreadOp.VERTICAL == SpreadOp.VERTICAL:
            features.append(np.array([Xt[1]]))
            features.append(np.array([Xt[1], self._invert(Xt[1])]))

        if C is not None:
            Ct = C.transpose()
            features.append(Ct)

        if self.spread_op_flag & SpreadOp.CENTROIDS == SpreadOp.CENTROIDS:
            X_c = np.array([np.sqrt(((X - c) ** 2).sum(1)) for c in self.centroids])

            if self.influence_max is not None and self.influence_max > 0:
                point_tree = spatial.cKDTree(self.centroids)
                nb_ixs = point_tree.query_ball_point(X, self.influence_max)
                test_data_influence_free_ixs = np.array(
                    list(map(lambda item: item[0] - test_start_ix, filter(lambda item: len(item[1]) == 0, enumerate(nb_ixs)))))
                # nb_ixs = list(map(lambda ix: set(ix), nb_ixs))
                # max_p = np.max(Xt, axis=1)
                # min_p = np.min(Xt, axis=1)
                # rect = max_p - min_p
                # dist = np.sqrt(rect ** 2).sum(0)
                # X_masked = np.full((len(self.centroids),X.shape[0]), -dist)
                # for i in range(test_start_ix, len(X)):
                #     for j in range(len(self.centroids)):
                #         if j not in nb_ixs[i - test_start_ix]:
                #             X_masked[j][i] = dist
                # Xtcf = np.maximum(X_c, X_masked)
                features.append(X_c)
            else:
                features.append(X_c)
                test_data_influence_free_ixs = []

        if self.spread_op_flag & SpreadOp.NEIGHBORS == SpreadOp.NEIGHBORS:
            if self.influence_max is not None and self.influence_max > 0:
                if nb_ixs is None:
                    point_tree = spatial.cKDTree(self.centroids)
                    nb_ixs = point_tree.query_ball_point(X, self.influence_max)
                    test_data_influence_free_ixs = np.array(
                        list(map(lambda item: item[0] - test_start_ix, filter(lambda item: len(item[1]) == 0, enumerate(nb_ixs)))))
                nn_features = np.zeros((3, len(nb_ixs)))
                y_val = np.min(self.values)
                for i in range(0, len(nb_ixs)):
                    if len(nb_ixs[i]) == 0:
                        nn_features[0][i] = 0
                        nn_features[1][i] = 0
                        nn_features[2][i] = y_val
                    else:
                        neighbors = self.centroids[nb_ixs[i]]
                        if X_c is not None:
                            dst = X_c[nb_ixs[i]]
                        else:
                            dst = np.sqrt(((neighbors - X[i]) ** 2).sum(1))
                        dst = np.maximum(dst, 1)
                        inv_dst = 1 / dst
                        sum_inv_dst = np.sum(inv_dst)
                        norm_dst = inv_dst / sum_inv_dst
                        nn_features[0][i] = np.min(norm_dst) # min dst
                        nn_features[1][i] = np.average(norm_dst)# avg dst
                        vals = self.values[nb_ixs[i]]
                        nn_features[2][i] = np.sum(vals * norm_dst) # avg val
            else:
                if X_c is None:
                    X_c = np.array([np.sqrt(((X - c) ** 2).sum(1)) for c in self.centroids])
                dst = X_c
                dst = np.maximum(dst, 1)
                min_f = np.min(dst) # min dst
                avg_f = np.average(dst)  # avg dst
                sum_dst = np.sum(dst)
                norm_dst = dst / sum_dst
                avg_val = np.sum(self.vals * norm_dst)
                nn_features = np.concatenate((min_f, avg_f, avg_val), axis = 0)
            features.append(nn_features)

        Xtf = np.concatenate(features, axis = 0)
        Xf = Xtf.transpose()
        #normalize
        mns = Xf.min(axis=0)
        mxs = Xf.max(axis=0)
        Xf = Xf - mns[np.newaxis, :]
        dsts = mxs - mns
        Xf = Xf / dsts[np.newaxis,:]

        return Xf, test_data_influence_free_ixs

    def _check(self, grid_points, auxiliary_grid_values):
        return grid_points, auxiliary_grid_values

    def _augment_corners(self, X_train, y_train, X_test):
        xy_max = np.amax(X_test, axis = 0)
        xy_min = np.amin(X_test, axis = 0)
        X_augmented = np.array([
            [xy_min[0], xy_min[1]],
            [xy_min[0], xy_max[1]],
            [xy_max[0], xy_min[1]],
            [xy_max[0], xy_max[1]]
        ])
        point_tree = spatial.cKDTree(self.centroids)
        xx, yy = point_tree.query(X_augmented)
        y_augmented = y_train[yy]
        new_X_train = np.concatenate((X_train, X_augmented), axis = 0)
        new_y_train = np.concatenate((y_train, y_augmented), axis = 0)
        return new_X_train, new_y_train

    def execute(self, grid_points, auxiliary_grid_values = None):
        # Check that X and y have correct shape
        grid_points, auxiliary_grid_values = self._check(grid_points, auxiliary_grid_values)
        # union X data
        X_train = self.centroids
        y_train = self.values
        X_test = grid_points
        if (self.spread_op_flag & SpreadOp.VERTICAL == self.spread_op_flag.VERTICAL) \
            or (self.spread_op_flag & SpreadOp.HORIZONTAL == self.spread_op_flag.HORIZONTAL):
            X_train, y_train = self._augment_corners(X_train, y_train, X_test)
        X_data = np.concatenate((X_train, X_test), axis = 0)
        # union aux data
        C_train = self.auxiliary_values
        C_test = auxiliary_grid_values
        if C_train is not None and C_test is not None:
            C_data = np.concatenate((C_train, C_test), axis = 0)
        else:
            C_data = None
        # union values
        def_val = min(self.values)
        X_data, test_data_influence_free_ixs = self._transform(X_data, len(X_train), C_data)
        # prepare training and validation datasets
        X_train = X_data[:len(X_train)]
        X_test = X_data[len(X_train):]
        if self.eval_ratio is not None:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.eval_ratio, \
                                                          random_state=self.train_eval_split_random_state)
        #build keras model
        self.model = Sequential()
        self.model.add(Dense(self.fcl0, input_dim=X_train.shape[1], kernel_initializer='he_uniform', activation='relu',))
        self.model.add(Dropout(rate=self.dropout))
        self.model.add(BatchNormalization())
        self.model.add(Dense(self.fcl1, activation='relu'))
        self.model.add(Dropout(rate=self.dropout))
        self.model.add(Dense(self.fcl2, activation='relu', \
                              kernel_regularizer=regularizers.l1_l2(l1=self.regularizer_l1, l2=self.regularizer_l2),
                              bias_regularizer=regularizers.l1_l2(l1=self.regularizer_l1, l2=self.regularizer_l2),
                              activity_regularizer=regularizers.l2(self.regularizer_l2)
                             ))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
        #training
        callbacks = [GetBest(monitor='loss', verbose=1, mode='min')]
        if self.eval_ratio is not None:
            self.learn_history = self.model.fit(X_train, y_train, epochs=self.n_epochs, batch_size=self.batch_size, \
                       validation_data=(X_val, y_val), callbacks=callbacks, verbose=self.verbose_mode)
        else:
            self.learn_history = self.model.fit(X_train, y_train, epochs=self.n_epochs, batch_size=self.batch_size, \
                       callbacks=callbacks, verbose=self.verbose_mode)
        #inference
        y_test = self.model.predict(X_test, workers=self.inference_workers, use_multiprocessing=True)
        mval = np.min(y_train)

        if test_data_influence_free_ixs is not None and len(test_data_influence_free_ixs) > 0:
            y_test[test_data_influence_free_ixs] = mval
        # Return the predicted values
        return y_test
