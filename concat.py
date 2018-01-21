"""
.. todo::

    * simplify / streamline ``core`` DataFrame routines in light of current algorithm
    * attach plot.concat to Concatenator
    * experiment with different smoothing parameters for the spline
    * use of previously removed dataseries (dispensables) for the offset confidence calculation
    * check other possibilities for confidence of offsets, e.g.
        * R^2 / generalized OLS ideas

Other remarks
-------------
    * If the dbscan outlier routine goes haywire, an alternative to choosing the cluster with the most members could be to fit a regression to each cluster and chose the one with the slope closest to one.

"""

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
    """
    Usage example::

        cc = Concatenator(directory='data/4/1')

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, data_flag=None, time_adj=0, **kwargs)
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

        v = self.time_zone(var.iloc[:, order]).resample('30T').asfreq()
        self.delta = v.index.freq.delta.to_timedelta64()
        self.pre_screen(v.dropna(0, 'all'), dispensable)

        end_points = self.var.end_points
        self.starts = self.var.index.get_indexer(end_points.loc['start'])
        self.ends = self.var.index.get_indexer(end_points.loc['end'])

        self.out = pd.DataFrame(np.nan, index=self.var.index, columns=['resid', 'extra', 'interp', 'outliers', 'concat'])
        self.file_names = self.var.columns.get_level_values('filename')
        self.long_short = self.var.columns.get_level_values('long_short')
        self.traverse()
        self.concat()

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

    def distance(self, start, stop):
        d = (stop.values.reshape((1, -1)) - start.values.reshape((-1, 1))).astype('timedelta64[s]').astype(float)
        D = np.abs(d)
        return np.where(D < D.T, d, d.T) + np.diag(np.repeat(-np.inf, len(start)))

    # remember, this returns the expected indexes -1, because this fits the need of some of the parts of the algorithm
    @staticmethod
    def _contiguous(s):
        ds = np.diff(s.astype(int))
        i = np.where(ds)[0]
        if len(i) == 0:
            return np.array([(0, len(s))])
        return np.pad(i, ((ds[i[[0, -1]]] == [-1, 1]).astype(int)), 'linear_ramp', end_values = [0, len(s)]).reshape((-1, 2))

    @staticmethod
    def contiguous(s):
        """Returns the indexes between which the input array is ``True`` or ``1`` as closed intervals in the rows of the returnd :class:`~numpy.ndarray` (i.e. for integer-based slicing, the second index has to be incremented by one, and the length of the interval is equal to the difference along dimension one **+1**).
        """
        ds = np.diff(s.astype(int))
        starts = np.pad(np.where(ds==1)[0] + 1, (int(s[0]==1), 0), 'constant')
        ends = np.pad(np.where(ds==-1)[0], (0, int(s[-1]==1)), 'constant', constant_values=len(ds))
        return np.vstack((starts, ends)).T

    # this is the outlier detection routine, using DBSCAN on either the differences (overlap) or residuals (spline)
    def dbscan(self, resid, return_labels=False, contiguous=False, eps=2):
        db = DBSCAN(eps=eps).fit(resid.reshape((-1, 1)))
        labels, counts = np.unique(db.labels_, return_counts=True)
        l = labels[counts.argmax()] == db.labels_
        if contiguous:
            j = self.contiguous(l)
            a, b = j[np.diff(j, 1, 1).argmax(), :]
            l = slice(a, b+1)
        return (l, [db.labels_ == i for i in sorted(labels)]) if return_labels else l

    # orthogonal distance regression
    def odr(self, i, plot=False, eps=2):
        c = self.var.iloc[:, i:i+2].dropna(0, 'any')

        # outliers are detected from the differenc between the two series
        diff = c.diff(1, 1).values[:, 1]
        k, labels = self.dbscan(diff, True, True, eps)

        x, y = c.iloc[k].values.T
        o = odr.ODR(odr.Data(x, y), odr.models.unilinear, beta0=[1, 0]).run()
        diff = diff[k]
        offs = np.nanmean(diff)
        m = np.argmin(abs(diff - offs)) + np.where(k)[0].min()
        self.starts[i+1] += m
        self.ends[i] = self.starts[i+1] - 1

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            t = c.index.values.astype('datetime64[s]').astype(float).reshape((-1, 1))
            x, y = c.values.T
            for h in labels:
                pl = axs[0].plot(t[h], x[h], 'x')[0]
                axs[0].plot(t[h], y[h], '*', color=pl.get_color())
                axs[1].plot(x[h], y[h], 'x')
                axs[1].plot(o.xplus, o.y, 'r-')
            axs[1].plot(x, y, 'mo', mfc='none')
            axs[1].yaxis.tick_right()
            plt.show()
        else:
            idx = c.index.symmetric_difference(c.index[k])
            j = np.where(np.diff(idx) != self.delta)[0]
            if len(j) > 1:
                print('Complex outlier structure deteced at knot {}'.format(i))
            elif len(j) == 1: # means there are outliers on either end of the overlap
                self.out.loc[idx[:j[0]+1], 'outliers'] = i + 1
                self.out.loc[idx[j[0]+1:], 'outliers'] = i
            elif len(idx) > 0: # means there are outliers only in one of the two series
                self.out.loc[idx, 'outliers'] = i + int(idx[0] == c.index[0])
            return {
                'offs': offs,
                'kind': 'mean',
                'stdev': diff.std() / np.sqrt(len(diff)),
                'odr_slope': o.beta[0],
                'odr_offs': o.beta[1],
                'RSS1': np.mean(o.delta ** 2),
                'RSS2': np.mean(o.eps ** 2),
            }

    def spline(self, i, plot=False, smooth = 10., eps=2, half_length=20):
        m = int(np.ceil((self.starts[i+1] + self.ends[i]) / 2))
        j = np.arange(m - half_length, m + half_length + 1)

        c = self.var.iloc[j, i:i+2].sum(1, skipna=True) # this is a hack, only works if there is no overlap of course!
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
            l = c.index[self.dbscan(sp.resid, eps=eps)]
            o = c.index.symmetric_difference(l)
            sp.fit(np.ma.MaskedArray(c, c.index.isin(o)), smooth, m - j[0])
            self.out.ix[j, 'extra'] = sp.spline
            self.out.ix[j, 'resid'] = sp.resid
            if len(o) > 0:
                outliers = np.where(self.var.ix[o, i:i+2].notnull())[1]
                self.out.loc[o, 'outliers'] = i + np.unique(outliers).item()
                self.starts[i+1] = self.var.index.get_loc(max(o)) + 1
                self.ends[i] = self.var.index.get_loc(min(o)) - 1 # ``starts`` and ``ends`` give the actual indexes, not the slice arguments
            return {
                'offs': sp.offset,
                'kind': 'spline',
                'stdev': sp.resid.std() / np.sqrt(len(l)),
                'orig_stdev' : orig_stdev
            }

    def traverse(self, smooth=10.):
        offsets = pd.DataFrame()

        # first pass over all 'knots', offset computations
        for i, (b, c) in enumerate(zip(self.starts[1:], self.ends[:-1])):
            # if there's a gap or a flush connection
            if b - c > 0:
                offs = self.spline(i, smooth=smooth)

            # if there is overlap
            else:
                offs = self.odr(i)

            offs.update({
                'conn': '{}-{}'.format(*self.long_short[i:i+2]),
                'diff': self.var.ix[self.starts[i+1], i+1] - self.var.ix[self.ends[i], i]
            })
            offsets = offsets.append(offs, ignore_index=True)

        # second pass over contiguous sequences of 'short' time series - offset *corrections*
        cols = offsets.columns.tolist()
        idx = offsets.columns.get_indexer(['offs', 'diff'])
        cols.extend(['corr_offs', 'diff_csum'])
        c = self.contiguous(self.long_short == 'short')
        corr_offs = []
        for a, b in c:
            a = max(a-1, 0)
            offs = offsets.iloc[a: b+1]
            if a == 0 and self.long_short[0] == 'short':
                offs = pd.concat((offs, offs.iloc[::-1, idx].cumsum().iloc[::-1]), 1)
                offs.index = self.file_names[a: b+1]
            else:
                csum = - offs.iloc[:, idx].cumsum()
                if self.long_short[b + 1] == 'long':
                    corr = offs.stdev
                    corr = - corr / corr.sum() * csum['offs'].iloc[-1]
                    csum = pd.concat((csum['offs'] + corr, csum['diff']), 1)
                offs = pd.concat((offs, csum), 1)
                offs.index = self.file_names[a+1: b+2]
            offs.columns = cols
            corr_offs.append(offs)

        self.offsets = pd.concat((corr_offs), keys=range(c.shape[0])).T

    def concat(self, no_offset=[], smooth=5., half_length=20):
        """Perform the actual concatenation.

        :param no_offset: list of indexes of the columns in the ordered :attr:`var` :class:`LogFrame` whose computed offset should be skipped (i.e. set to zero)
        :param smooth: smoothing spline smoothing parameter for spline-based interpolation of the missing values

        """
        for i, (a, b) in enumerate(zip(self.starts, self.ends)):
            offs = 0.
            if self.long_short[i] == 'short' and i not in no_offset:
                offs = self.offsets.loc['corr_offs'].xs(self.file_names[i], level=1).item()
            self.out.ix[a: b+1, 'concat'] = self.var.iloc[a: b+1, i] + offs

        for i, (b, c) in enumerate(zip(self.starts[1:], self.ends[:-1])):
            if b - c > 1:
                m = int(np.ceil((b + c) / 2))
                j = np.arange(m - half_length, m + half_length + 1)
                mask = self.out.ix[j, 'outliers'].notnull()
                sp = Bspline(j)
                sp.fit(np.ma.MaskedArray(self.out.ix[j, 'concat'], mask), smooth)
                self.out.ix[j, 'interp'] = sp.spline
                self.out['concat'] = self.out['concat'].where(self.out['concat'].notnull(), self.out['interp'])

    # doesn't work if gaps aren't infilled!
    def ar(self, i, plot=False, ar=1, half_length=500):
        a, b = self.contiguous(self.long_short == 'short')[i]
        a = max(a-1, 0)

        knots = self.knots[a: b+1]
        j = np.arange(max(0, knots[0] - half_length), min(self.var.shape[0]+ 1, knots[-1] + half_length + 1))
        concat = self.out.ix[j, 'concat']
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

