import pandas as pd
import numpy as np
from datetime import datetime
from scipy.sparse import csgraph
import os
from warnings import catch_warnings, simplefilter
from .core import Reader
from .bspline import Bspline


class Concatenator(Reader):
    def __init__(self, logdir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # while testing only for self.level
        # this removes the outlying data series for now (although we might want to use it in the end)
        # var = self.level.data.drop('AK4_LL-203_temp_August20_2012', 1, 'filename')
        var = self.level.data

        # only self.var is sorted correctly, i.e. in the same way as self.columns and self.tf_data
        self.var = var.resample('30T').asfreq().organize_time()
        self.meta = self.var.apply(self.phase, 0).to_frame('ft')
        self.meta['dm'] = np.argmax(self.var.groupby(self.var.index.hour).mean().values, 0)
        self.meta['mean'] = self.var.mean()
        self.meta['std'] = self.var.std()

        self.var['extra'] = np.nan
        self.var['concat'] = np.nan
        self.var['resid'] = np.nan
        self.var['outliers'] = np.nan

        # just so it isn't recomputed every time (it's a @property)
        self.end_points = self.var.end_points

        # self.traverse()
        # self.meta = self.meta.iloc[self.order]



    @staticmethod
    def phase(x, p=86400):
        x = x.dropna()
        t = x.index.values.astype('datetime64[s]').astype(float)
        N = len(x)
        a = np.sum(np.exp(-1j * 2 * np.pi * t / p) * x.values)
        phase = np.arctan2(np.imag(a), np.real(a))
        cycle = a * np.exp (1j * 2 * np.pi * t / p) / N
        return phase

    def overlap_and_contained(self):
        """Returns symmetric matrix of which series overlap with each other:
            * a 1 entry means that the row/columns combo overlaps - the row/column indexes refer to the numeric index of a series in the original data.
            * The `contained` :class:`DataFrame` has a 1 for series that are entirely contained 
in another one.

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


    @staticmethod
    def outliers(resid, thresh):
        r = np.repeat(resid.reshape((1, -1)), len(resid), 0) + np.diag(np.empty_like(resid.flatten()) * np.nan)
        r = np.nansum((np.abs(r) > thresh * np.nanstd(r, 1)).astype(int), 0)
        return None if np.sum(r) == 0 else r.astype(bool)

    def traverse(self, thresh=3, smooth=10., stop=None):
        # needs to be called within a graph.as_default() context
        self.start = self.var.index.get_indexer(self.end_points.loc['start'])
        self.stop = self.var.index.get_indexer(self.end_points.loc['stop'])

        D = self.distance(self.start, self.stop)

        with catch_warnings():
            simplefilter('ignore')
            _, p = csgraph.dijkstra(-D, return_predecessors=True)

        i, c = np.unique(p, return_counts=True)
        first, last = sorted(i[c == 1], key=lambda j: self.stop[j])
        # first = self.start.argsort()[0]
        # self.order = [self.stop.argsort()[-1]]
        self.order = [last]
        self.knots = []
        offsets = []

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
                # k = np.r_[c - 2 + np.arange(0, -3, -1) * n, b + 2 + np.arange(3) * n]
                # k.sort()
                # x = np.arange(k[0], k[-1]+1)
                x = np.arange(m - 20, m + 21)
                y1 = self.var.iloc[x[0]: m, i[0]].values.copy() # IMPORTANT! if not copy, value will be set on original
                y2 = self.var.iloc[m: x[-1]+1, i[1]].values.copy()
                sp = Bspline(x)
                sp.fit2(y1, y2, smooth)
                j = self.outliers(sp.resid, thresh)
                if j is not None:
                    self.var.ix[np.where(j)[0] + x[0], 'outliers'] = 1.
                    y1[j[:len(y1)]] = np.nan # here value will be set to NaN on original DataFrame, apparently
                    y2[j[len(y1):]] = np.nan
                    sp.fit2(y1, y2, smooth)

                offs = sp.offset
                kind = 'spline'
                resid = np.nanmean(sp.resid ** 2)
                self.var.ix[x, 'extra'] = sp.spline
                self.var.ix[x, 'resid'] = sp.resid
            else:
                y1 = self.var.iloc[b: c+1, i[0]]
                y2 = self.var.iloc[b: c+1, i[1]]
                offs = np.nanmean(y2 - y1)
                kind = 'mean'
                resid = np.nanmean((y2 - offs - y1) ** 2)
                self.start[i[1]] = m
                self.stop[i[0]] = m - 1

            l = self.var.columns[i].get_level_values(0)
            fn = self.var.columns[i].get_level_values('filename')
            if l[1] == 'short':
                if len(offsets) > 0:
                    offsets[0]['start'] = (-offs, resid, kind)
                else:
                    offsets.insert(0, {'file': fn[1], 'start': (-offs, resid, kind)})
            if l[0] == 'short':
                offsets.insert(0, {'file': fn[0], 'end': (offs, resid, kind)})
                if l[1] == 'short':
                    offsets[0]['next'] = fn[1]

            if stop is not None:
                print(kind)
                if kind == 'spline':
                    return sp

        for i in self.order:
            a = self.start[i]
            b = self.stop[i] + 1
            self.var.ix[a: b, 'concat'] = self.var.iloc[a: b, i]

        self.offsets = []
        while len(offsets) > 0:
            o = [offsets.pop(0)]
            try:
                while o[-1]['next'] == offsets[0]['file']:
                    o.append(offsets.pop(0))
            except:
                pass
            finally:
                self.offsets.append(o)

    def files_in_order(self):
        fn = self.var.columns.get_level_values('filename')[self.order]
        for o, f in zip(self.order, fn):
            print('{}: {}'.format(o, f))
