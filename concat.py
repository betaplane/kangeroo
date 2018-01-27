import pandas as pd
import numpy as np
from scipy.sparse import csgraph
import scipy.odr as odr
from sklearn.covariance import MinCovDet
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import DBSCAN
from statsmodels.api import tsa
import matplotlib.pyplot as plt
import os
from warnings import catch_warnings, simplefilter
from .core import Reader
from .bspline import Bspline


class Concatenator(Reader):
    """
    Usage example::

        cc = Concatenator(directory='data/4/1', var='level)

    """
    def __init__(self, directory=None, variable=None, resample='30T', correct_time=False, dispensable_thresh=3600, copy=None):
        """
        Usage example::

            cc = Concatenator(directory='data/4/1', var='level)

        :param var: 
        :param directory: 
        :param resample: 
        :param correct_time:
        :param dispensable_thresh: 
        :param copy: 
        :returns: 
        :rtype: 

        """
        super().__init__(directory, copy, variable, resample)

        end_points = self.var.apply(self.get_start_end)
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
            if d > -dispensable_thresh:
                dispensable.append(len(order) - 2)

        v = self.time_zone(self.var.iloc[:, order], correct_time).resample(resample).asfreq()
        self.delta = v.index.freq.delta.to_timedelta64()
        self.var = self.pre_screen(v, dispensable)
        self.out = pd.DataFrame(np.nan, index=self.var.index, columns=['resid', 'extra', 'interp', 'outliers', 'concat'])

        self.traverse()
        self.concat()
        print('')

    def prepare(self):
        end_points = self.var.apply(self.get_start_end)
        self.starts = self.var.index.get_indexer(end_points.loc['start'])
        self.ends = self.var.index.get_indexer(end_points.loc['end'])
        self.file_names = self.var.columns.get_level_values('file')
        self.long_short = self.var.columns.get_level_values('length')

    def time_zone(self, var, correct):
        if correct is False:
            return pd.concat((var, ), 1, keys=[0], names=['time_adj'] + var.columns.names)
        elif isinstance(correct, list):
            a = var.iloc[: correct]
            b = var.drop(a.columns, 1)
        else:
            phase = var.apply(self.phase, 0)
            i = np.arange(var.shape[1]).reshape((-1, 1))
            tr = DecisionTreeRegressor(max_leaf_nodes=2).fit(i, phase)
            cl = tr.apply(i)
            a = var.iloc[:, cl == 1]
            b = var.iloc[:, cl == 2]
        a.index = a.index + pd.Timedelta(5, 'h')
        print("\nThe following files' timestamps have been changed by 5 hours:\n")
        for f in a.columns.get_level_values('file'):
            print(f)
        return pd.concat((a, b), 1, keys=[5, 0], names=['time_adj'] + var.columns.names)


    def pre_screen(self, var, disp, thresh=10):
        """Uses Minimum Covariance Determinand / Mahalanobis distance ideas to detect outliers, loosely based on :cite:`chawla_k-means:_2013`.

        """
        fx = var.columns.names.index('file')
        feat = pd.concat((var.mean(), var.std()), 1)
        mcd = MinCovDet().fit(feat)
        md = mcd.mahalanobis(feat)
        s = set(np.where(md > thresh)[0])
        k = s.intersection(disp).union(s.intersection({0, var.shape[1]}))
        self.dispensable = list(set(disp) - k)
        if len(k) > 0:
            print('\n\nThe following files have been removed from the concatenation as unnecessary outliers:\n')
        for i in k:
            print(var.columns[i][fx])
        return var.drop(var.columns[list(k)], axis=1)

    @staticmethod
    def phase(x, p=86400):
        x = x.dropna()
        t = x.index.values.astype('datetime64[s]').astype(float)
        N = len(x)
        a = np.sum(np.exp(-1j * 2 * np.pi * t / p) * x.values)
        phase = np.arctan2(np.imag(a), np.real(a))
        cycle = a * np.exp (1j * 2 * np.pi * t / p) / N
        return phase

    @staticmethod
    def distance(start, stop):
        d = (stop.values.reshape((1, -1)) - start.values.reshape((-1, 1))).astype('timedelta64[s]').astype(float)
        D = np.abs(d)
        return np.where(D < D.T, d, d.T) + np.diag(np.repeat(-np.inf, len(start)))

    @staticmethod
    def contiguous(s):
        """Returns the indexes between which the input array is ``True`` or ``1`` as closed intervals in the rows of the returnd :class:`~numpy.ndarray` (i.e. for integer-based slicing, the second index has to be incremented by one, and the length of the interval is equal to the difference along dimension one **+1**).
        """
        ds = np.diff(s.astype(int))
        starts = np.pad(np.where(ds==1)[0] + 1, (int(s[0]==1), 0), 'constant')
        ends = np.pad(np.where(ds==-1)[0], (0, int(s[-1]==1)), 'constant', constant_values=len(ds))
        return np.vstack((starts, ends)).T

    # this is the outlier detection routine, using DBSCAN on either the differences (overlap) or residuals (spline)
    def dbscan(self, resid, return_labels=False, contiguous=False, masked=False, eps=2):
        x = resid[~resid.mask] if masked else resid
        db = DBSCAN(eps=eps).fit(x.reshape((-1, 1)))
        labels, counts = np.unique(db.labels_, return_counts=True)
        l = labels[counts.argmax()] == db.labels_
        if masked: # this is only relevant for the spline overlaps
            L = np.zeros(len(resid)).astype(bool)
            L[~resid.mask] = l
            l = L
        if contiguous: # this is only relevant for the odr overlaps
            j = self.contiguous(l)
            a, b = j[np.diff(j, 1, 1).argmax(), :]
            l = slice(a, b+1)
        return (l, [db.labels_ == i for i in sorted(labels)]) if return_labels else l

    # orthogonal distance regression
    def odr(self, i, plot=False, eps=2):
        c = self.var.iloc[:, i:i+2].dropna(0, 'any')

        # outliers are detected from the differenc between the two series
        diff = c.diff(1, 1).values[:, 1]
        k, labels = self.dbscan(diff, True, True, eps=eps)

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

    def spline(self, i, plot=False, smooth = 10., eps=2, pad=20):
        m = int(np.ceil((self.starts[i+1] + self.ends[i]) / 2))
        j = np.arange(m - pad, m + pad + 1)

        c = self.var.iloc[j, i:i+2].sum(1, skipna=True) # this is a hack, only works if there is no overlap of course!
        sp = Bspline(j)
        sp.fit(c, smooth)
        orig_stdev = sp.resid.std() / np.sqrt(len(j))

        labels = self.dbscan(sp.resid, eps=eps, return_labels=plot, masked=True)
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            t = c.index
            for k, h in enumerate(labels[1]):
                axs[0].plot(t[h], c.loc[h], 'o', color=colors[k])
                axs[1].plot(t[h], sp.resid[h], 'o', color=colors[k])
            axs[0].plot(t, sp.spline, 'x-', color=colors[k+1])
            plt.show()
        else:
            o = c.index.symmetric_difference(c.index[labels])
            sp.fit(np.ma.MaskedArray(c, ~labels), smooth, m - j[0])
            self.out.ix[j, 'extra'] = sp.spline
            self.out.ix[j, 'resid'] = sp.resid
            if len(o) > 0:
                row, col = np.where(self.var.ix[o, i:i+2].notnull())
                if len(row) > 0: # could be a gap
                    self.out.loc[o[row], 'outliers'] = i + col
                self.starts[i+1] = self.var.index.get_loc(max(o)) + 1
                self.ends[i] = self.var.index.get_loc(min(o)) - 1 # ``starts`` and ``ends`` give the actual indexes, not the slice arguments
            return {
                'offs': sp.offset,
                'kind': 'spline',
                'stdev': sp.resid.std() / np.sqrt(labels.astype(int).sum()),
                'orig_stdev' : orig_stdev
            }

    def traverse(self, smooth=10.):
        self.prepare()
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
            })
            offsets = offsets.append(offs, ignore_index=True)

        # second pass over contiguous sequences of 'short' time series - offset *corrections*
        cols = offsets.columns.tolist()
        idx = offsets.columns.get_indexer(['offs'])
        cols.extend(['corr_offs'])
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
                if self.var.shape[1] == b + 2 and self.long_short[b + 1] == 'long':
                    corr = offs.stdev.values.reshape((-1, 1))
                    corr = - corr / corr.sum() * csum.iloc[-1].item()
                    csum = csum + corr
                offs = pd.concat((offs, csum), 1)
                offs.index = self.file_names[a+1: b+2]
            offs.columns = cols
            corr_offs.append(offs)

        self.offsets = pd.concat((corr_offs), keys=range(c.shape[0])).T

    def print_offsets(self, columns=['corr_offs']):
        """Print the offset table in slightly more readable format, with the indexes that can be used for the parameter ``no_offset`` in :meth:`concat` in column ``idx`` and the 'transition zone' index as the left-most level in the hierarchical index, which can be used for the parameter ``use_spline`` in :meth:`concat`. Columns to be printed can be given as kwarg ``columns``.
        """
        offs = self.offsets.T
        offs['idx'] = range(offs.shape[0])
        b = offs['conn'] != 'short-long'
        if offs['conn'].iloc[0] != 'long-long':
            b[0] = True
        offs = offs[b]
        print(offs[['idx']+columns])

    def concat(self, no_offset=[], use_spline=[], smooth_spline=5., pad_spline=20, slope_tol=0.2):
        """Perform the actual concatenation.

        :param no_offset: list of indexes of columns in the ordered :attr:`var` DataFrame whose computed offset should be skipped (i.e. set to zero)
        :type no_offset: :obj:`list`
        :param use_spline: list of top-level indexes in the :attr:`offsets` DataFrame (corresponding to the colored sections of the plot produced by :meth:`plot`) to which spline-based interpolation should be applied
        :type use_spline: :obj:`list`
        :param smooth_spline: smoothing parameter for spline-based interpolation of the missing values
        :type smooth_spline: :obj:`float``
        :param pad_spline: Padding, in integer indexes, on either side of the transition zone across which a smoothing spline should be fit (roughly equivalent to parameter ``pad`` in :meth:`spline`)
        :type pad_spline: :obj:`int`
        :param slope_tol: deviation from a slope of 1 in the regression of two series in an overlap region which is tolerated without printing a message
        :type slope_tol: :obj:`float`

        """
        no_offs = self.offsets.columns[no_offset].get_level_values('file')
        for i, (a, b) in enumerate(zip(self.starts, self.ends)):
            fn = self.file_names[i]
            offs = 0.
            if self.long_short[i] == 'short':
                c = self.offsets.xs(fn, 1, 'file', drop_level=False)
                if fn not in no_offs:
                    offs = self.offsets.loc['corr_offs', c.columns].item()
                else:
                    self.offsets.loc['corr_offs', c.columns] = 0.
            self.out.ix[a: b+1, 'concat'] = self.var.iloc[a: b+1, i] + offs

        for i, (b, c) in enumerate(zip(self.starts[1:], self.ends[:-1])):
            if b - c > 1:
                m = int(np.ceil((b + c) / 2))
                j = np.arange(m - pad_spline, m + pad_spline + 1)
                mask = self.out.ix[j, 'outliers'].notnull()
                sp = Bspline(j)
                sp.fit(np.ma.MaskedArray(self.out.ix[j, 'concat'], mask), smooth_spline)
                self.out.ix[j, 'interp'] = sp.spline
                self.out['concat'] = self.out['concat'].where(self.out['concat'].notnull(), self.out['interp'])

        if len(use_spline) == 0 and 'odr_slope' in self.offsets.index:
            s = self.offsets.loc['odr_slope']
            c = s[np.abs(s - 1) > slope_tol].index.get_level_values(0)
            print('\n\nThe following transitions have slope abnormalities:\n')
            for i in c:
                print(i, self.offsets[i].columns.tolist())
        else:
            for i in use_spline:
                j = self.file_names.get_indexer(self.offsets[i].columns)
                start = self.var.iloc[:, j[0]].dropna().index.min()
                end = self.var.iloc[:, j[-2]].dropna().index.max()
                start, end = self.var.index.get_indexer([start, end]) + pad_spline * np.array([-1, 1])
                if j[0] > 0:
                    j = np.r_[j[0]-1 , j]
                # from IPython.core.debugger import Tracer;Tracer()()
                b = self.out['outliers'].values.reshape((-1, 1)) != j.reshape((1, -1))
                x = self.var.iloc[:, j].where(b, np.nan).iloc[start: end+1, :]
                y = pd.concat([c for _, c in x.iteritems()], 0).dropna()
                sp = Bspline(y.index)
                sp.fit(y, smooth_spline)
                z = pd.Series(sp.spline, index=y.index).drop_duplicates()
                self.out.loc[z.index, 'interp'] = z
                self.out.loc[z.index, 'concat'] = z

    def merge_info(self):
        cols = self.var.columns.to_frame(False)
        cols['start'] = self.var.index[self.starts]
        cols['end'] = self.var.index[self.ends]
        offs = self.offsets.T.reset_index()[['file','corr_offs']]
        cols = cols.merge(offs, on='file', how='outer')
        cols.loc[cols['length']=='long', 'corr_offs'] = 0.
        var = self.var.copy()
        var.columns = pd.MultiIndex.from_tuples(list(cols.as_matrix()))
        var.columns.names = cols.columns
        return var

    def to_csv(self):
        if hasattr(self, 'old_var'):
            old_files = self.old_var.columns.get_level_values('file')
            new_files = self.var.columns.get_level_values('file')
            intsec = old_files.intersection(new_files)
            f = old_files.get_indexer(intsec)
            old_var = self.old_var.drop(intsec, 1, level='file')
            self.starts[0] = self.var.index.get_loc(self.old_var.columns.get_level_values('start')[f[0]])
            var = pd.concat((old_var, self.merge_info()), 1).dropna(0, 'all')
            self.out = self.out.iloc[self.starts[0]:].combine_first(self.old_out)
        else:
            var = self.merge_info().dropna(0, 'all')

        out_path = os.path.join(self.directory, 'out')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        var.to_csv(os.path.join(out_path, '{}_input.csv'.format(self.variable)))
        self.out.to_csv(os.path.join(out_path, '{}_output.csv'.format(self.variable)))


    # autoregressive fit - not used currently + doesn't work if gaps aren't infilled yet!
    def ar(self, i, plot=False, ar=1, pad=500):
        a, b = self.contiguous(self.long_short == 'short')[i]
        a = max(a-1, 0)

        knots = self.knots[a: b+1]
        j = np.arange(max(0, knots[0] - pad), min(self.var.shape[0]+ 1, knots[-1] + pad + 1))
        concat = self.out.ix[j, 'concat']
        a = tsa.AR(concat).fit(ar)
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

    def plot(self):
        """Produce a helpful plot to inspect the concatenated timeseries.
        """
        fig, ax = plt.subplots()
        idx = self.var.index
        plt.plot(idx, self.out['concat'])
        plt.plot(idx, self.out['extra'], 'rx-')
        plt.plot(idx, self.out['interp'], 'gx-')

        fn = self.var.columns.names.index('file')
        le = self.var.columns.names.index('length')
        height = self.var.columns.get_level_values('length').str.contains('short').astype(int).sum()
        o = self.out['outliers']

        j = 0
        for i, c in enumerate(self.var.iteritems()):
            y = c[1].dropna()
            if c[0][le] == 'long':
                plt.plot(y, 'k-')
            else:
                start_t = y.index[0]
                p = plt.plot(y)[0]

                ax.axvspan(idx[self.starts[i]], idx[self.ends[i]], alpha=.4, facecolor=p.get_color())

                ax.annotate(c[0][fn], xy=(start_t, 1 - (1 + j) / height),
                            xycoords=('data', 'axes fraction'), color=p.get_color())
                j += 1

            outliers = c[1][o == i].dropna()
            plt.plot(outliers, 'mo', mfc='none')

        fig.show()
