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
from .core import Reader


class Optimizer(Reader):

    tf_dtype = tf.float64
    """Default TensorFlow_ data type to use."""

    time_units = 'datetime64[s]'
    time_delta = 'timedelta64[s]'

    def __init__(self, logdir=None, *args, **kwargs):
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

        self.traverse()

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
        d = (stop.reshape((1, -1)) - start.reshape((-1, 1)))
        D = np.abs(d)
        return np.where(D < D.T, d, d.T) + np.diag(np.repeat(-np.inf, len(start)))

    def ar_resid(self, concat):
        x0 = tf.reshape(concat[:-1], (-1, 1))
        x1 = tf.reshape(concat[1:], (-1, 1))
        lsq = tf.matrix_solve_ls(x0, x1)
        return tf.reshape(x0 * lsq - x1, (-1,))

    def optimize(self, learn=0.1, n_iter=100):
        prog = tf.keras.utils.Progbar(n_iter)
        with self.graph.as_default():
            op = tf.train.AdamOptimizer(learn).minimize(self.loss)
            tf.global_variables_initializer().run(session=self.sess)
            for i in range(n_iter):
                _, l = self.sess.run([op, self.loss])
                prog.update(i, [('Loss', l)])


    def traverse(self, radius=100, thresh=15):
        # needs to be called within a graph.as_default() context

        filenames = self.var.columns.get_level_values('filename')

        self.start = self.var.index.get_indexer(self.end_points.loc['start'])
        self.stop = self.var.index.get_indexer(self.end_points.loc['stop'])

        D = self.distance(self.start, self.stop)
        _, p = csgraph.dijkstra(-D, return_predecessors=True)
        first = self.start.argsort()[0]
        self.order = [self.stop.argsort()[-1]]
        self.knots = []
        losses = []

        s = 0
        while self.order[0] != first:
            self.order.insert(0, p[first, self.order[0]])
            i = self.order[:2]
            a, b = self.start[i]
            c, d = self.stop[i]

            # should always be the center (beginning of second in case of flush connection)
            m = int(np.ceil((b + c) / 2))
            self.knots.insert(0, m)

            # if there's a gap or a flush connection
            if b - c > 0:
                
            else:
                loss = self.tf_data[b: c+1, i[0]] - self.tf_data[b: c+1, i[1]]
                # print('searching overlap between files {} and {}'.format(*filenames[i]))
                # for j in range(b, c):
                #     concat = tf.concat((self.tf_data[a: j, i[0]], self.tf_data[j: d+1, i[1]]), 0)
                #     resid.append(tf.reduce_sum(self.ar_resid(concat) ** 2))
                # m = b + tf.argmin(resid).eval(session=self.sess)
                concat = tf.concat((self.tf_data[a: m, i[0]], self.tf_data[m: d+1, i[1]]), 0)
                self.start[i[1]] = m
                self.stop[i[0]] = m - 1

            losses.append(loss ** 2)

            if s==4: break
            s += 1

        self.loss = tf.concat(losses, 0)

    @property
    def concat(self):
        idx = [np.arange(a, b+1) for a, b in zip(self.start, self.stop)]
        data = [tf.gather(self.tf_data[:, i], j) for i, j in enumerate(idx)]
        k = self.extra_idx
        idx.append(k)
        data.append(tf.gather(self.extra_var, k))
        return tf.dynamic_stitch(idx, data)

    @property
    def extra_idx(self):
        a = self.start[self.order[1:]]
        b = self.stop[self.order[:-1]] + 1
        d = (a > b)
        return np.hstack(np.arange(*z) for z in zip(b[d], a[d]))


    def setup(self, learn=0.01, logdir=None):
        with self.graph.as_default():
            # extra_loss = tf.reduce_sum(tf.gather(self.raw_weights, self.extra_idx, axis=1), name='extra_loss')
            loss = self.ar_loss + self.extra_loss * 100
            # loss = self.extra_loss ** 4
            self.step = tf.get_variable('global_step', initializer=0, trainable=False)
            # self.train = tf.train.GradientDescentOptimizer(learn)
            self.train = tf.train.AdamOptimizer(learn)
            train_op = self.train.minimize(loss, self.step)

            self.sess = tf.Session(graph=self.graph)

            tf.global_variables_initializer().run(session=self.sess)
            self.original_concat = self.concat.eval(session=self.sess)

            if logdir is not None:
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
            self.sess.run(tf.assign(self.knots, tf.clip_by_value(self.knots, self.k_lims[0, :], self.k_lims[1, :])))
            prog.update(i, [('Loss', l)])

        if hasattr(self, 'tb_writer'):
            self.tb_writer.flush() # IMPORTANT! will not work without .close() or .flush()

    def __del__(self):
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
