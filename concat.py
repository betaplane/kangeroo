import pandas as pd
import numpy as np
from scipy.sparse import csgraph
import scipy.odr as odr
from sklearn.covariance import MinCovDet
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
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

        self.traverse()


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

    def dbscan(self, resid, return_labels=False, contiguous=False, eps=2):
        db = DBSCAN(eps=eps).fit(resid.reshape((-1, 1)))
        labels, counts = np.unique(db.labels_, return_counts=True)
        l = labels[counts.argmax()] == db.labels_
        if contiguous:
            j = self.contiguous(l) + 1 # the + 1 used to be inside contiguous()
            l = slice(*j[np.diff(j, 1, 1).argmax(), :])
        return (l, [db.labels_ == i for i in sorted(labels)]) if return_labels else l

    # orthogonal distance regression
    def odr(self, i, plot=False, eps=2):
        c = self.var.iloc[:, i:i+2].dropna(0, 'any')

        # outliers are detected from the differenc between the two series
        diff = c.diff(1, 1).values[:, 1]
        k, labels = self.dbscan(diff, True, True, eps)

        x, y = c.iloc[k].values.T
        self.concat.loc[c.index.symmetric_difference(c.index[k]), 'outliers'] = -2
        o = odr.ODR(odr.Data(x, y), odr.models.unilinear, beta0=[1, 0]).run()
        diff = diff[k]
        offs = np.nanmean(diff)
        m = np.argmin(abs(diff - offs)) + np.where(k)[0].min()
        self.starts[i+1] += m
        self.ends[i] = self.starts[i+1] - 1

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            t = c.index.values.astype('datetime64[s]').astype(float).reshape((-1, 1))
            for h in labels:
                pl = axs[0].plot(t[h], x[h], 'x')[0]
                axs[0].plot(t[h], y[h], '*', color=pl.get_color())
                axs[1].plot(x[h], y[h], 'x')
                axs[1].plot(o.xplus, o.y, 'r-')
            axs[1].plot(x, y, 'mo', mfc='none')
            axs[1].yaxis.tick_right()
            plt.show()
        else:
            return {
                'offs': offs,
                'kind': 'mean',
                'stdev': diff.std() / np.sqrt(len(diff)),
                'odr_slope': o.beta[0],
                'odr_offs': o.beta[1],
                'RSS1': np.mean(o.delta ** 2),
                'RSS2': np.mean(o.eps ** 2),
            }

    def spline(self, i, m, plot=False, smooth = 10., eps=2, half_length=20):
        j = np.arange(m - half_length, m + half_length + 1)

        c = self.var.iloc[j, i:i+2].sum(1, skipna=True)
        sp = Bspline(j)
        sp.fit(c, smooth)
        orig_stdev = sp.resid.std() / np.sqrt(len(j))

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            t = c.index
            for k, l in enumerate(sorted(labels)):
                h = db.labels_ == l
                axs[0].plot(t[h], c.loc[h], 'o', color=colors[k])
                axs[1].plot(t[h], sp.resid[h], 'o', color=colors[k])
            axs[0].plot(t, sp.spline, 'x-', color=colors[k+1])
            plt.show()
        else:
            l = self.dbscan(sp.resid)
            o = c.index.symmetric_difference(c.index[l])
            sp.fit(np.ma.MaskedArray(c, ~l), smooth, m - j[0])
            self.concat.ix[j, 'extra'] = sp.spline
            self.concat.ix[j, 'resid'] = sp.resid
            if len(o) > 0:
                self.concat.loc[o, 'outliers'] = -1
                self.starts[i+1] = self.var.index.get_loc(max(o)) + 1
                self.ends[i] = self.var.index.get_loc(min(o)) - 1 # ``starts`` and ``ends`` give the actual indexes, not the slice arguments
            return {
                'offs': sp.offset,
                'kind': 'spline',
                'stdev': sp.resid.std() / np.sqrt(len(j)),
                'orig_stdev' : orig_stdev
            }

    def traverse(self, smooth=10.):
        self.knots = []
        offsets = pd.DataFrame()
        file_names = self.var.columns.get_level_values('filename')
        long_short = self.var.columns.get_level_values('long_short')

        for i, (b, c) in enumerate(zip(self.starts[1:], self.ends[:-1])):
            # should always be the center (beginning of second in case of flush connection)
            m = int(np.ceil((b + c) / 2))

            # if there's a gap or a flush connection
            if b - c > 0:
                offs = self.spline(i, m, smooth=smooth)

            # if there is overlap
            else:
                offs = self.odr(i)
                m = self.starts[i+1]

            self.knots.append(m)
            offs.update({
                'conn': '{}-{}'.format(*long_short[i:i+2]),
                'diff': self.var.ix[self.starts[i+1], i+1] - self.var.ix[self.ends[i], i]
            })
            offsets = offsets.append(offs, ignore_index=True)

        cols = offsets.columns.tolist()
        idx = offsets.columns.get_indexer(['offs', 'diff'])
        cols.extend(['corr_offs', 'diff_csum'])
        corr_offs = []
        c = self.contiguous(long_short == 'short')
        self.corr_offs = []
        for a, b in c:
            offs = offsets.iloc[a: b+1]
            if a == 0 and long_short[0] == 'short':
                offs = pd.concat((offs, offs.iloc[::-1, idx].cumsum().iloc[::-1]), 1)
                offs.index = file_names[a: b+1]
            else:
                csum = - offs.iloc[:, idx].cumsum()
                if long_short[b + 1] == 'long':
                    corr = offs.stdev
                    corr = - corr / corr.sum() * csum['offs'].iloc[-1]
                    csum = pd.concat((csum['offs'] + corr, csum['diff']), 1)
                offs = pd.concat((offs, csum), 1)
                offs.index = file_names[a+1: b+2]
            offs.columns = cols
            corr_offs.append(offs)

        self.offsets = pd.concat((corr_offs), keys=range(c.shape[0])).T
        corr_offs = self.offsets.loc['corr_offs']

        for i, (a, b) in enumerate(zip(self.starts, self.ends)):
            offs = corr_offs.xs(file_names[i], level=1).item() if long_short[i] == 'short' else 0
            self.concat.ix[a: b+1, 'concat'] = self.var.iloc[a: b+1, i] + offs

    # doesn't work if gaps aren't infilled!
    def ar(self, i, plot=False, ar=1, half_length=500):
        long_short = self.var.columns.get_level_values('long_short')
        a, b = self.contiguous(long_short == 'short')[i]

        knots = self.knots[a: b+1]
        j = np.arange(max(0, knots[0] - half_length), min(self.var.shape[0]+ 1, knots[-1] + half_length + 1))
        concat = self.concat.ix[j, 'concat']
        a = sm.tsa.AR(concat).fit(ar)
        r = [a.resid[self.var.index[k]] for k in knots]
        print(r)

        if plot:
            fig, ax = plt.subplots()
            plt.plot(concat)
            for k in knots:
                ax.axvline(self.var.index[k], color='g')

            bx = ax.twinx()
            bx.plot(a.resid, 'r')
            plt.show()
        else:
            return r

