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
    * clean up new overlap_fraction / contained check
* test learning rate etc.
* maybe map the 'chain' order more directly to columns instead of ints
* write a separate summary for each transition / offset
* deal with transitions without / with too short overlaps (unclipped outliers affect outcome)
* make plot.all / flagged more robust w.r.t. time_ranges when .data only is passed

"""
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from scipy.sparse import csgraph
import os
from .core import *



class Optimizer(Reader):

    tf_dtype = tf.float64
    """Default TensorFlow_ data type to use."""

    time_units = 'datetime64[s]'
    time_delta = 'timedelta64[s]'

    bridge_length = 100

    def __init__(self, thresh=3600, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # while testing only for self.level
        # this removes the outlying data series for now (although we might want to use it in the end)
        var = self.level.data.drop('AK4_LL-203_temp_August20_2012', 1, 'filename')

        # only self.var is sorted correctly, i.e. in the same way as self.columns and self.tf_data
        self.var = var.resample('30T').asfreq().organize_time()

        # just so it isn't recomputed every time (it's a @property)
        self.end_points = self.var.end_points

        # self.start_sorted = self.var.columns.get_indexer(self.end_points.loc['start'].sort_values().index)
        # self.end_sorted = self.var.columns.get_indexer(self.end_points.loc['stop'].sort_values().index)

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('data'):
                self.offsets = tf.get_variable('offsets', (1, self.var.short.shape[1]), self.tf_dtype,
                                               tf.zeros_initializer)
                short = tf.constant(self.var.short.fillna(0).values, self.tf_dtype, name='short') + self.offsets
                long = tf.constant(self.var.long.fillna(0).values, self.tf_dtype, name='long')
                concat = tf.concat((long, short), 1, name='data')
            self.extend_data(concat, 100, thresh)


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

    def overlap_fraction(self):
        """
        Returns a matrix of the fraction of a series' duration to the shorter of the two possible overlap distances (i.e. stop - start with both combinations of two series).
            * **If an element is > 1**, its **row index** refers to the series which is **entirely contained** in the other series, while its **column index** refers to the series within which it is **contained**.

        """
        stop = self.end_points.loc['stop'].values.reshape((1, -1))
        start = self.end_points.loc['start'].values.reshape((1, -1))
        # duration = (stop - start).astype('datetime64[s]')
        m = (stop - start.T).astype('timedelta64[s]')
        return np.where(m < m.T, m, m.T).astype(float) + np.diag(np.repeat(-np.inf, self.var.shape[1]))
        # self.overlap = m * m.T
        # self.contained = np.where(overlap > duration) # I believe the second is contained in the first - CHECK!!!

    def distance(self, start, stop):
        d = (stop.reshape((1, -1)) - start.reshape((-1, 1))).astype(self.time_delta)
        D = np.abs(d)
        return np.where(D < D.T, d, d.T).astype(float) + np.diag(np.repeat(-np.inf, len(start)))

    def add_variable(self, start, stop, i):
        n = self.bridge_length // 2
        m = self.bridge_length - n
        t = start[i[2]] + (start[i[2]] - stop[i[0]]) / 2
        j = self.var.index.get_loc(t, 'nearest')
        ext = tf.get_variable('intp_{}'.format(i[1]), (self.bridge_length, ), self.tf_dtype,
                              tf.constant_initializer(np.nanmean(self.var)))
        v = tf.concat((tf.zeros(j - n, self.tf_dtype), ext, tf.zeros(self.var.shape[0] - j - m, self.tf_dtype)), 0)
        a = self.var.index.values[j - n]
        b = self.var.index.values[j - n + self.bridge_length]
        return v, a, b

    def extend_data(self, tf_data, length, thresh=3600):
        # needs to be called within a graph.as_default() context
        stop = self.end_points.loc['stop'].values
        start = self.end_points.loc['start'].values

        D = self.distance(start, stop)
        _, p = csgraph.dijkstra(-D, return_predecessors=True)
        first = start.argsort()[0]
        i = [stop.argsort()[-1]]
        extra_vars = []
        joins = []
        with tf.variable_scope('extra_vars'):
            while i[0] != first:
                i.insert(0, p[first, i[0]])
                if D[i[0], i[1]] < thresh:
                    i.insert(1, self.var.shape[1] + len(extra_vars))
                    v, a, b = self.add_variable(start, stop, i)
                    extra_vars.append(v)
                    start = np.r_[start, a]
                    stop = np.r_[stop, b]

        self.idx_map = dict(zip(i, range(len(i))))
        self.extra_idx = [self.idx_map[i] for i in range(self.var.shape[1], len(i))]

        dt = self.var.index.freq.delta.asm8.astype(self.time_delta).astype(float) * 10
        start = start.astype(self.time_units).astype(float) / dt
        stop = stop.astype(self.time_units).astype(float) / dt
        m = ((start + stop) / 2)[i]
        self.k_init = (np.vstack((m[:-1], start[i[1:]])).max(0) + np.vstack((stop[i[:-1]], m[1:])).min(0)) / 2
        with tf.variable_scope('knots'):
            self.knots = tf.get_variable('knots', (len(self.k_init),), self.tf_dtype, tf.constant_initializer(self.k_init))
            l = tf.reshape(tf.concat((tf.cast([start[first]], self.tf_dtype),
                           tf.reshape(tf.stack((self.knots, self.knots), 1), (-1,)),
                           tf.cast([stop[i[-1]]], self.tf_dtype)), 0), (-1, 2))

        with tf.variable_scope('weights'):
            self.slope = tf.get_variable('slope', (), self.tf_dtype, tf.constant_initializer(dt), trainable=False)
            t = tf.cast(np.array(self.var.index, dtype=self.time_units, ndmin=2).T.astype(float),
                        self.tf_dtype, name='time') / dt
            self.raw_weights = self.slope * (t - l[:, 0]) * (t - l[:, 1]) / (l[:, 0] - l[:, 1])
            self.weights = tf.nn.softmax(self.raw_weights)
        with tf.variable_scope('concat'):
            data = tf.gather(tf.concat((tf_data, tf.stack(extra_vars, 1)), 1), i, axis=1)
            self.concat = tf.reduce_sum(self.weights * data, 1, name='concat')

        with tf.variable_scope('loss'):
            x0 = tf.reshape(self.concat[:-1], (-1, 1))
            x1 = tf.reshape(self.concat[1:], (-1, 1))
            lsq = tf.matrix_solve_ls(x0, x1)
            y = x0 * lsq
            self.resid = x0 * lsq - x1
            self.ar_loss = tf.reduce_sum(self.resid ** 2, name='ar_loss')
            lx = tf.gather(l, self.extra_idx, axis=0)
            # self.extra_loss = tf.reduce_mean(self.ar_loss * tf.reduce_sum(tf.nn.softmax(lx), 1))
            self.extra_loss = tf.reduce_sum((lx[:, 1] - lx[:, 0]), name='extra_loss')


    def setup(self, learn=0.01, logdir=None):
        with self.graph.as_default():
            # extra_loss = tf.reduce_sum(tf.gather(self.raw_weights, self.extra_idx, axis=1), name='extra_loss')
            loss = self.ar_loss + self.extra_loss
            # loss = self.extra_loss ** 4
            self.step = tf.get_variable('global_step', initializer=0, trainable=False)
            # self.train = tf.train.GradientDescentOptimizer(learn)
            self.train = tf.train.AdamOptimizer(learn)
            train_op = self.train.minimize(loss, self.step)

            self.sess = tf.Session(graph=self.graph)

            tf.global_variables_initializer().run(session=self.sess)
            self.original_concat = self.concat.eval(session=self.sess)

            if logdir is not None:
                subdir = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                self.tb_writer = tf.summary.FileWriter(os.path.join(logdir, subdir), graph=self.graph)
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('ar_loss', self.ar_loss)
                tf.summary.scalar('extra_loss', self.extra_loss)
                tf.summary.histogram('offsets', self.offsets)

                for i in range(len(self.k_init)):
                    tf.summary.scalar('knots', self.knots[i] - self.k_init[i])

                summary = tf.summary.merge_all()

                def update():
                    _, l, s, step = self.sess.run([train_op, loss, summary, self.step])
                    self.tb_writer.add_summary(s, step)
                    return l

            else:
                def update():
                    return self.sess.run([train_op, loss])

        self._update = update


    def update(self, steps):
        prog = tf.keras.utils.Progbar(steps)
        for i in range(steps):
            l = self._update()
            prog.update(i, [('Loss', l)])

        if hasattr(self, 'tb_writer'):
            self.tb_writer.flush() # IMPORTANT! will not work without .close() or .flush()

    def __del__(self):
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
