"""
Usage Example
=============
.. code-block:: python

    log = Log('data/4/1/')
    c = log.optimize()

Currently the variable log.level is used to develop.

TODO
====

* in :meth:`chains`, make gap jump criterion the next (non-NaN) timestamp in the original dataframe instead the next starting value

"""
import pandas as pd
import numpy as np
from statsmodels.tsa import ar_model
import tensorflow as tf
from core import *



class Optimizer(Reader):

    tf_dtype = tf.float32
    """Default TensorFlow_ data type to use."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        var = self.level.data  # while testing only for self.level:

        var = var.resample('30T').asfreq()

        self.shape = var.shape
        self.long_idx, self.short_idx = var.organize_time()

        # only self.var is sorted correctly, i.e. in the same way as self.columns and self.tf_data
        self.var = var.iloc[:, np.r_[self.long_idx, self.short_idx]]
        self.columns = self.var.columns

        self.end_points = self.var.end_points
        self.index = self.var.index

        self.start_sorted = self.columns.get_indexer(self.end_points.loc['start'].sort_values().index)
        self.end_sorted = self.columns.get_indexer(self.end_points.loc['stop'].sort_values().index)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.offsets = tf.get_variable('offsets', self.short_idx.shape, self.tf_dtype,
                                           tf.random_normal_initializer(mean=0, stddev=20))
            # data = tf.placeholder(self.tf_dtype, self.shape)
            data = tf.constant(var.fillna(0).values, self.tf_dtype)
            short = tf.gather(data, self.short_idx, axis=1) + self.offsets
            self.tf_data = tf.concat((tf.gather(data, self.long_idx, axis=1), short), 1)


    @classmethod
    def overlap(cls, s):
        """Returns symmetric matrix of which series overlap with each other:
            * a 1 entry means that the row/columns combo overlaps - the row/column indexes refer to the numeric index of a series in the original data.
            * The `contained` :class:`DataFrame` has a 1 for series that are entirely contained in another one.

        **Input**: a .data or .flag :class:`DataFrame` with :meth:`get_start_stop` applied to axis 0 (see :meth:`chains`).

        """
        # s = df.apply(cls.get_start_stop)
        stop = s.loc['stop'].values.reshape((1, -1)).astype(float)
        start = s.loc['start'].values.reshape((-1, 1)).astype(float)
        m = (stop - start)
        overlap = ((m > 0) & (m.T > 0)).astype(int) - np.diag(np.ones(m.shape[0]))
        contained = (((stop - stop.T) > 0) & ((start - start.T) > 0)).astype(int)
        return overlap, contained

    @staticmethod
    def overlap_fraction(s):
        """
        Returns a matrix of the fraction of a series' duration to the shorter of the two possible overlap distances (i.e. stop - start with both combinations of two series).
            * **If an element is > 1**, its **row index** refers to the series which is **entirely contained** in the other series, while its **column index** refers to the series within which it is **contained**.

        """
        stop = s.loc['stop'].values.reshape((1, -1))
        start = s.loc['start'].values.reshape((1, -1))
        duration = stop - start
        m = (stop - start.T)
        m[m.astype(float) < 0] = np.datetime64('nat')
        m = np.where(m < m.T, m, m.T)
        return m / duration.T

        # d = np.where(duration < duration.T, duration, duration.T)
        # m = np.where(m < d, m, d)
        # m1, m2 = m / duration, m / duration.T
        # return np.where(m1 > m2, m1, m2) - np.diag(np.ones(m.shape[0]))

    def chains(self, overlap, contained, exclude_contained=True):
        """Attempt at a method that finds all permutations of data series combinations based on overlaps. Returns a :class:`~numpy.array` where each row corresponds to a sequence of series indexes referring to the columns of the input :class:`DataFrame`. Also keeps track of gaps that needed to be jumped for each of the rows. (The routine is based on actual timestamps, so overlap and jumping gaps refers to actual time gaps.) See :meth:`chain_gangs` for example of invocation.
        """

        idx = self.start_sorted
        # will hold all the gaps that needed to be jumped because there is no overlap; keys are rows in the returned np.array
        gaps = {}

        def stack(k, row):
            i = row[-1]

            # this is the set of previous series that we will not allow the sequence to go back to - *unless* we are at a series entirely contained within another
            if exclude_contained:
                uturn = set(row).union(np.where(contained[:, i])[0]) # if we leave out all contained series, we remove them from the series we are jumping **to** (column index if we use row index otherwise)
            else:
                uturn = set(row) - set(np.where(contained[i, :])[0])

            n = list(set(np.where(overlap[i])[0]) - uturn) if i >= 0 else []

            # if there is a gap or we're at the end
            if len(n) == 0:
                j = np.where(idx == i)[0]
                if j + 1 < len(idx):
                    n = self.start_sorted[j + 1] # CHANGE THIS EVENTUALLY!!!!!!!!!
                    if k in gaps:
                        gaps[k].append((i, n.item()))
                    else:
                        gaps[k] = [(i, n.item())]
                # if we're at the end
                else:
                    n = [-9]
            return np.hstack((row.reshape((1, -1)).repeat(len(n), 0), np.array(n).reshape((-1, 1))))

        c = idx[0].reshape((1, 1))
        stop = self.end_sorted[-1]
        while not np.all([(stop in i) for i in c]):
            c = np.vstack([stack(*i) for i in enumerate(c)])

        print(gaps)
        return c

    def softmax(self, chain, overlap, s=1.):
        t = np.array(self.index, dtype='datetime64[s]', ndmin=2).astype(np.float32).T
        end_points = self.end_points.ix[['start', 'stop'], chain].values.astype('datetime64[s]')

        # this shifts the start of one series over the stop of the previous one
        cross = end_points.flatten()[1:-1].reshape((2, -1))
        # this finds the mid point of either the transition or the gap (if there's no overlap)
        self.mid_overlaps = (cross[0, :] + np.diff(cross, 1, 0) / 2).astype(float)

        init = tf.cast(self.mid_overlaps, self.tf_dtype)

        # with self.graph.as_default():
        self.cross = tf.get_variable('cross', dtype=self.tf_dtype, initializer=init)

        # this is the inflexion point halfway between the transition mid-points
        flex = self.cross[:, :-1] + (self.cross[:, 1:] - self.cross[:, :-1]) / 2

        # now I pad the ends appropriately
        cr = tf.concat(([[0.]], self.cross), 1)
        fl = tf.concat((cr[:, 1:2] / 2, flex), 1)

        # this constructs a 'triangular' function with zero crossings in the right directions at the `cross` points
        # (as arguments to the softmax function)
        x = tf.concat((fl - cr[:, :-1] - np.abs(t - fl), t - cr[:, -1:]), 1)
        self.weights = tf.nn.softmax(s * x, 1)
        concat = self.weights * tf.gather(self.tf_data, chain, axis=1)
        self.concat = tf.reduce_sum(concat, 1)

        # this computes an AR(1) model, the residuals of which are used as the loss function
        # see also :meth:`cut_ends`
        x0 = tf.reshape(self.concat[:-1], (-1, 1))
        x1 = tf.reshape(self.concat[1:], (-1, 1))
        lsq = tf.matrix_solve_ls(x0, x1)
        y = x0 * lsq
        loss = tf.losses.mean_squared_error(x1, x0 * lsq)
        return loss

    def optimize(self, learn=0.01, steps=500, chains_only=False):
        # get overlaps and contained
        overlap, contained = self.overlap(self.end_points)

        chains = self.chains(overlap, contained)
        if chains_only:
            return chains

        with self.graph.as_default():
            loss = self.softmax(chains[0], overlap)
            opt = tf.train.GradientDescentOptimizer(learn).minimize(loss)

        prog = tf.keras.utils.Progbar(steps)
        with tf.Session(graph=self.graph) as s:
            tf.global_variables_initializer().run(session=s)

            self.original_concat = self.concat.eval(session=s)

            for i in range(steps):
                out = s.run([opt, loss])
                prog.update(i)
            return s.run([self.offsets, self.cross])

    def initialize(self):
        """For debugging / development. Initializes TensorFlow_ variables and returns session."""
        sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            tf.global_variables_initializer().run(session = sess)
        return sess
