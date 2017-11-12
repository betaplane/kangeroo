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
* test learning rate etc.
* maybe map the 'chain' order more directly to columns instead of ints
* allow continuing of training ops
* write a separate summary for each transition / offset
* clip loss function to just the overlap areas (*not* the AR model)

"""
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
from .core import *



class Optimizer(Reader):

    tf_dtype = tf.float32
    """Default TensorFlow_ data type to use."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        var = self.level.data  # while testing only for self.level

        # only self.var is sorted correctly, i.e. in the same way as self.columns and self.tf_data
        self.var = var.resample('30T').asfreq().organize_time()

        # just so it isn't recomputed every time (it's a @property)
        self.end_points = self.var.end_points

        self.start_sorted = self.var.columns.get_indexer(self.end_points.loc['start'].sort_values().index)
        self.end_sorted = self.var.columns.get_indexer(self.end_points.loc['stop'].sort_values().index)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.offsets = tf.get_variable('short_offsets', (1, self.var.short.shape[1]), self.tf_dtype,
                                           tf.random_normal_initializer(mean=0, stddev=20))
            short = tf.constant(self.var.short.fillna(0).values, self.tf_dtype) + self.offsets
            long = tf.constant(self.var.long.fillna(0).values, self.tf_dtype)
            self.tf_data = tf.concat((long, short), 1)


    def overlap_and_contained(self):
        """Returns symmetric matrix of which series overlap with each other:
            * a 1 entry means that the row/columns combo overlaps - the row/column indexes refer to the numeric index of a series in the original data.
            * The `contained` :class:`DataFrame` has a 1 for series that are entirely contained in another one.

        **Input**: a .data or .flag :class:`DataFrame` with :meth:`get_start_stop` applied to axis 0 (see :meth:`chains`).

        """
        stop = self.end_points.loc['stop'].values.reshape((1, -1)).astype(float)
        start = self.end_points.loc['start'].values.reshape((-1, 1)).astype(float)
        m = (stop - start)
        self.overlap = ((m > 0) & (m.T > 0)).astype(int) - np.diag(np.ones(m.shape[0]))
        self.contained = (((stop - stop.T) > 0) & ((start - start.T) > 0)).astype(int)

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

    def chains(self, exclude_contained=True):
        """Attempt at a method that finds all permutations of data series combinations based on overlaps. Returns a :class:`~numpy.array` where each row corresponds to a sequence of series indexes referring to the columns of the input :class:`DataFrame`. Also keeps track of gaps that needed to be jumped for each of the rows. (The routine is based on actual timestamps, so overlap and jumping gaps refers to actual time gaps.) See :meth:`chain_gangs` for example of invocation.
        """
        if not hasattr(self, 'overlap'):
            self.overlap_and_contained()

        idx = self.start_sorted
        # will hold all the gaps that needed to be jumped because there is no overlap; keys are rows in the returned np.array
        gaps = {}

        def stack(k, row):
            i = row[-1]

            # this is the set of previous series that we will not allow the sequence to go back to - *unless* we are at a series entirely contained within another
            if exclude_contained:
                uturn = set(row).union(np.where(self.contained[:, i])[0]) # if we leave out all contained series, we remove them from the series we are jumping **to** (column index if we use row index otherwise)
            else:
                uturn = set(row) - set(np.where(self.contained[i, :])[0])

            n = list(set(np.where(self.overlap[i])[0]) - uturn) if i >= 0 else []

            # if there is a gap or we're at the end
            if len(n) == 0:
                j = np.where(idx == i)[0]
                if j + 1 < len(idx):
                    n = self.start_sorted[j + 1] # CHANGE THIS EVENTUALLY!!!!!!!!!
                    if k in gaps:
                        gap = self.end_points.ix['start', n.item()] - self.end_points.ix['stop', i]
                        gaps[k].append((i, n.item(), gap))
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

    def softmax(self, chain, s=1.):
        t = np.array(self.var.index, dtype='datetime64[s]', ndmin=2).astype(np.float32).T
        end_points = self.end_points.ix[['start', 'stop'], chain].values.astype('datetime64[s]')

        # this shifts the start of one series over the stop of the previous one
        cross = end_points.flatten()[1:-1].reshape((2, -1)).astype(float)

        # this finds the mid point of either the transition or the gap (if there's no overlap)
        self.mid_overlaps = (cross[0, :] + np.diff(cross, 1, 0) / 2)

        init = tf.cast(self.mid_overlaps, self.tf_dtype)

        # with self.graph.as_default():
        self.cross = tf.get_variable('cross', dtype=self.tf_dtype, initializer=init)
        self.cross_limit = tf.clip_by_value(self.cross,
                                            tf.cast(cross[0, :], self.tf_dtype), tf.cast(cross[1, :], self.tf_dtype))

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
        self.resid = x0 * lsq - x1
        return tf.reduce_sum(self.resid ** 2)

    def optimize(self, learn=0.01, steps=500, logdir=None):
        chains = self.chains()
        self.chain = chains[0] # we use only one chain for now

        with self.graph.as_default():
            loss = self.softmax(self.chain)
            opt = tf.train.GradientDescentOptimizer(learn).minimize(loss)

            prog = tf.keras.utils.Progbar(steps)
            self.sess = tf.Session(graph=self.graph)

            tf.global_variables_initializer().run(session=self.sess)
            self.original_concat = self.concat.eval(session=self.sess)

            if logdir is not None:
                subdir = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                self.tb_writer = tf.summary.FileWriter(os.path.join(logdir, subdir))

                tf.summary.scalar('loss', loss)
                tf.summary.histogram('offsets', self.offsets)
                tf.summary.histogram('transitions', self.cross - self.mid_overlaps)
                summary = tf.summary.merge_all()

                for i in range(steps):
                    _, l, s, _  = self.sess.run([opt, loss, summary, self.cross_limit])
                    self.tb_writer.add_summary(s, global_step=i)
                    prog.update(i, [('Loss', l)])

            else:
                for i in range(steps):
                    _, l  = self.sess.run([opt, loss])
                    prog.update(i, {'Loss': l})

            self.tb_writer.close() # IMPORTANT! will not work without .close() or .flush()

    def initialize(self):
        """For debugging / development. Initializes TensorFlow_ variables and returns session."""
        sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            tf.global_variables_initializer().run(session = sess)
        return sess
