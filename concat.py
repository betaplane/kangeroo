import pandas as pd
import numpy as np
from scipy.sparse import csgraph
import scipy.odr as odr
from sklearn.covariance import MinCovDet
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from warnings import catch_warnings, simplefilter
from .core import Reader
from .bspline import Bspline


class Concatenator(Reader):
    """Usage example::

            cc = Concatenator(directory='data/4/1')

    """
    def __init__(self, logdir=None, *args, **kwargs):
        super().__init__(*args, data_flag=None, time_adj=0, **kwargs)

        # while testing only for self.level
        # this removes the outlying data series for now (although we might want to use it in the end)
        # var = self.level.data.drop('AK4_LL-203_temp_August20_2012', 1, 'filename')

        # self.var = self.pre_screen(var.resample('30T').asfreq().organize_time())
        var = self.level.organize_time()

        end_points = var.end_points
        starts = end_points.loc['start']
        ends = end_points.loc['end']
        D = self.distance(starts, ends)

        with catch_warnings():
            simplefilter('ignore')
            _, p = csgraph.dijkstra(-D, return_predecessors=True)

        end = ends.argsort()[-1]
        start = starts.argsort()[0]
        order = [start, p[end, start]]

        dispensable = []
        while order[-1] != end:
            order.append(p[end, order[-1]])
            d = D[order[-1], order[-3]]
            if d > -3600:
                dispensable.append(len(order) - 2)

        self.pre_screen(
            self.time_zone(var.iloc[:, order]).resample('30T').asfreq().dropna(0, 'all'),
            dispensable)

        end_points = self.var.end_points
        self.starts = self.var.index.get_indexer(end_points.loc['start'])
        self.ends = self.var.index.get_indexer(end_points.loc['end'])

        self.concat = pd.DataFrame(np.nan, index=self.var.index, columns=['resid', 'interp', 'outliers', 'concat'])

        self.traverse(start=0, end=self.var.shape[1])


    def time_zone(self, var):
        phase = var.apply(self.phase, 0)
        i = np.arange(var.shape[1]).reshape((-1, 1))
        tr = DecisionTreeRegressor(max_leaf_nodes=2).fit(i, phase)
        cl = tr.apply(i)
        a = var.iloc[:, cl == 1]
        b = var.iloc[:, cl == 2]
        a.index = a.index + pd.Timedelta(5, 'h')
        a.columns = a.columns.set_levels([5], 'time_adj')
        print("The following files' timestamps have been changed by 5 hours:\n")
        for f in a.columns.get_level_values('filename'):
            print(f)
        print('')
        return pd.concat((a, b), 1)


    def pre_screen(self, var, disp, thresh=10):
        fx = var.columns.names.index('filename')
        feat = pd.concat((var.mean(), var.std()), 1)
        mcd = MinCovDet().fit(feat)
        md = mcd.mahalanobis(feat)
        s = set(np.where(md > thresh)[0])
        k = s.intersection(disp).union(s.intersection({0, var.shape[1]}))
        self.var = var.drop(var.columns[list(k)], axis=1)
        self.dispensable = list(set(disp) - k)
        for i in k:
            print('File {} removed from concatenation as unnecessary outlier.'.format(var.columns[i][fx]))

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
        d = (stop.values.reshape((1, -1)) - start.values.reshape((-1, 1))).astype('timedelta64[s]').astype(float)
        D = np.abs(d)
        return np.where(D < D.T, d, d.T) + np.diag(np.repeat(-np.inf, len(start)))

    @staticmethod
    def contiguous(s):
        ds = np.diff(s.astype(int))
        i = np.where(ds)[0]
        if len(i) == 0:
            return np.array([(0, len(s))])
        return np.pad(i, ((ds[i[[0, -1]]] == [-1, 1]).astype(int)), 'linear_ramp', end_values = [0, len(s)]).reshape((-1, 2))


    # orthogonal distance regression + DBSCAN clustering
    def odr(self, i, plot=False, eps=2):
        c = self.var.iloc[:, i:i+2].dropna(0, 'any')
        x, y = c.values.T
        t = c.index.values.astype('datetime64[s]').astype(float).reshape((-1, 1))
        # o = odr.ODR(odr.Data(x, y), odr.models.unilinear, beta0=[1, 0]).run()
        # r = np.abs(np.vstack((o.delta, o.eps))).max(0)
        # db = DBSCAN(eps=eps).fit(r.reshape((-1, 1)))

        db = DBSCAN(eps=eps).fit((y - x).reshape((-1, 1)))
        labels, counts = np.unique(db.labels_, return_counts=True)
        j = self.contiguous(labels[counts.argmax()] == db.labels_) + 1 # the + 1 used to be inside contiguous()
        k = slice(*j[np.diff(j, 1, 1).argmax(), :])

        xk, yk = x[k], y[k]
        self.concat.loc[c.index.symmetric_difference(c.index[k]), 'outliers'] = -2
        o = odr.ODR(odr.Data(xk, yk), odr.models.unilinear, beta0=[1, 0]).run()

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            for l in sorted(labels):
                h = db.labels_ == l
                pl = axs[0].plot(t[h], x[h], 'x')[0]
                axs[0].plot(t[h], y[h], '*', color=pl.get_color())
                axs[1].plot(x[h], y[h], 'x')
                axs[1].plot(o.xplus, o.y, 'r-')
            axs[1].plot(xk, yk, 'mo', mfc='none')
            axs[1].yaxis.tick_right()
            plt.show()
        else:
            return xk, yk, o

    @staticmethod
    def outliers(resid, thresh):
        r = np.repeat(resid.reshape((1, -1)), len(resid), 0) + np.diag(np.empty_like(resid.flatten()) * np.nan)
        r = np.nansum((np.abs(r) > thresh * np.nanstd(r, 1)).astype(int), 0)
        return None if np.sum(r) == 0 else r.astype(bool)

    def spline(self, i, plot=False, smooth = 10., eps=2, half_length=20):
        m = int(np.ceil((self.ends[i] + self.starts[i+1]) / 2))
        j = np.arange(m - half_length, m + half_length + 1)
        c = self.var.iloc[:, i:i+2].values.copy() # IMPORTANT! if not copy, value will be set on original
        x, y = c[j[0]: m, 0], c[m: j[-1]+1, 1]
        sp = Bspline(j)
        sp.fit(np.r_[x, y], smooth)
        db = DBSCAN(eps=eps).fit(abs(sp.resid).reshape((-1, 1)))
        labels, counts = np.unique(db.labels_, return_counts=True)
        l = labels[counts.argmax()]
        k = j[db.labels_ == l]
        o = self.var.index[j].symmetric_difference(self.var.index[k])
        self.concat.loc[o, 'outliers'] = -1
        c.iloc[k, :] = np.nan

        # if j is not None:
        #     self.concat.ix[np.where(j)[0] + x[0], 'outliers'] = -1.
        #     y1[j[:len(y1)]] = np.nan # here value will be set to NaN on original DataFrame, apparently
        #     y2[j[len(y1):]] = np.nan
        #     sp.fit2(y1, y2, smooth)

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            t = self.var.index[j]
            for k, l in enumerate(sorted(labels)):
                h = db.labels_ == l
                axs[0].plot(t[h], c[j[h], :], 'o', color=colors[k])
                axs[1].plot(t[h], sp.resid[h], 'o', color=colors[k])
            axs[0].plot(t, sp.spline, 'x-', color=colors[k+1])
        else:
            return outliers

    def traverse(self, start, end, thresh=3, smooth=10.):
        self.knots = []
        offsets = pd.DataFrame()
        long_short = self.var.columns.get_level_values('long_short')

        for i in range(start, end-1):
            a, b = self.starts[i: i+2]
            c, d = self.ends[i: i+2]

            # should always be the center (beginning of second in case of flush connection)
            m = int(np.ceil((b + c) / 2))
            self.knots.append(m)
            conn = '{}-{}'.format(*long_short[i:i+2])

            # if there's a gap or a flush connection
            if b - c > 0:
                # k = np.r_[c - 2 + np.arange(0, -3, -1) * n, b + 2 + np.arange(3) * n]
                # k.sort()
                # x = np.arange(k[0], k[-1]+1)
                x = np.arange(m - 20, m + 21)
                y1 = self.var.iloc[x[0]: m, i].values.copy() # IMPORTANT! if not copy, value will be set on original
                y2 = self.var.iloc[m: x[-1]+1, i+1].values.copy()
                sp = Bspline(x)
                sp.fit2(y1, y2, smooth)
                j = self.outliers(sp.resid, thresh)
                if j is not None:
                    self.concat.ix[np.where(j)[0] + x[0], 'outliers'] = -1.
                    y1[j[:len(y1)]] = np.nan # here value will be set to NaN on original DataFrame, apparently
                    y2[j[len(y1):]] = np.nan
                    sp.fit2(y1, y2, smooth)

                offsets = offsets.append({
                    'offs': sp.offset,
                    'kind': 'spline',
                    'resid': np.nanmean(sp.resid ** 2),
                    'conn': conn
                }, ignore_index=True)
                self.concat.ix[x, 'extra'] = sp.spline
                self.concat.ix[x, 'resid'] = sp.resid

            # if there is overlap
            else:
                y1, y2, o = self.odr(i)
                offs = np.nanmean(y2 - y1)
                offsets = offsets.append({
                    'offs': offs,
                    'kind': 'mean',
                    'resid': np.nanmean((y2 - offs - y1) ** 2),
                    'odr_slope': o.beta[0],
                    'odr_offs': o.beta[1],
                    'RSS1': np.mean(o.delta ** 2),
                    'RSS2': np.mean(o.eps ** 2),
                    'conn': conn
                }, ignore_index=True)
                self.starts[i+1] = m
                self.ends[i] = m - 1

        self.corr_offs = []
        for a, b in self.contiguous(long_short == 'short'):
            offs = offsets.ix[a: b, 'offs']
            if (a == 0) and long_short[0] == 'short':
                self.corr_offs.extend(np.cumsum(offs[::-1])[::-1])
            else:
                self.corr_offs.extend(offs[:-1] - offs.mean())
        self.offsets = offsets

        for i, (a, b) in enumerate(zip(self.starts, self.ends)):
            offs = self.corr_offs.pop(0) if long_short[i] == 'short' else 0
            self.concat.ix[a: b+1, 'concat'] = self.var.iloc[a: b+1, i] - offs
