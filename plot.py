"""
The methods of the :class:`Plots` class are accessible from :class:`~.LogFrame` instances via the :class:`~.LogFrame.plot` accessor.

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def get_time_ranges(column):
    out = {0: [np.nan, np.nan]}
    col = column.dropna().sort_index()
    d = col.diff()
    starts = d[d==1].dropna().index
    # the last valid timestamp is the one **before** the -1 diff location
    stops = d.index[d.index.get_indexer(d[d==-1].dropna().index) - 1]
    # end points of flag array become start / stop if their values are 1
    c = col.iloc[[0, -1]]
    end_points = c[c==1].dropna().index
    a = np.array(starts.append(stops).append(end_points).sort_values()).reshape((-1, 2))
    out.update({i: j for i, j in enumerate(a)})
    return pd.Series(out)


def cut_ends(df, std=3, lag=1, residuals=False):
    d = df.dropna(0, 'all').swaplevel(0, 'data_flag', 1)
    flags = d.xs('flag', 1, 'data_flag').copy()
    x = d.xs('data', 1, 'data_flag')
    t = np.array(x.index, dtype='datetime64')
    r = ar_model.AR(x, t).fit(lag).resid
    c = r.sort_values()[[0, -1]]
    cut_points = c[c.abs() > std * r.std()]
    if not cut_points.empty:
        # the occurence of the residual spike is delayed by lag
        idx = r.index.get_indexer(cut_points.sort_index().index) + lag
        for i in idx:
            if np.abs(d.shape[0] - i) < i:
                flags.iloc[i:, :] = 0
            else:
                flags.iloc[:i, :] = 0
        d = pd.concat((x, flags), 1, keys=['data', 'flag']).swaplevel(0, -1, 1)
        d.columns.names = df.columns.names
        df = d
    return (df, r) if residuals else df


class Plots(object):
    def __init__(self, df):
        self.df = df

    def all(self, flags=None, right=0.8, flagged=True, cut_ends=False):
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=right)

        # need to sort the series first by start time
        if flags is None:
            time_ranges = self.df.time_ranges
        else:
            flags.apply(get_time_ranges)

        sorted_index = time_ranges.loc[0].apply(lambda c:c[0]).sort_values().index
        names = sorted_index.get_level_values('filename')
        indexer = time_ranges.columns.get_indexer(sorted_index)
        height = len(names)

        used_or_not = {True:[], False:[]}
        for i, name in enumerate(names):
            d = self.df.xs(name, 1, 'filename')
            p = self.flagged(d, ax, cut_ends=cut_ends, residuals=False)
            u = False
            try:
                spans = time_ranges.xs(name, 1, 'filename')
                for _, span in spans.iterrows():
                    a, b = span.item()
                    ymin = 1-(1+i)/height
                    ax.axvspan(a, b, ymax=1-i/height, ymin=ymin, alpha=.4, facecolor=p.get_color())
                    ax.annotate(indexer[i], xy=(b, ymin), xycoords=('data', 'axes fraction'), color=p.get_color())
                    u = True
            except:
                pass
            used_or_not[u].append(Patch(color = p.get_color(), label='{}: {}'.format(indexer[i], name)))

        ax.add_artist(
            plt.legend(handles = used_or_not[False], loc='lower left', bbox_to_anchor=(1, 0), title='not used'))
        plt.legend(handles = used_or_not[True], bbox_to_anchor=(1, 1), title='used')

        fig.show()


    @staticmethod
    def flagged(df, ax=None, cut_ends=False, residuals=True, **kwargs):
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [None]

        df = df.dropna(0, 'all')
        if cut_ends:
            df, resid = cut_ends(df, residuals=True, **kwargs)
        data = df.xs('data', 1, 'data_flag')

        (fig, ax) = plt.subplots() if ax is None else (ax.figure, ax)
        p = ax.plot(data, color=colors[0])
        try:
            flags = df.xs('flag', 1, 'data_flag')
            ax.plot(data[(flags.isnull()) | (flags==0)], 'or')
            ax.plot(data.loc[[i for j in get_time_ranges(flags) for i in j]], 'og')
        except:
            pass
        if residuals:
            bx = ax.twinx()
            bx.plot(resid, color=colors[1])
        if ax is None:
            fig.show()
        else:
            return p[0]


def concat(optimizer):
    fig, ax = plt.subplots()
    plt.plot(optimizer.var.index, optimizer.concat.eval(session=optimizer.sess))

    short_idx = optimizer.var.columns.droplevel(0).get_indexer(optimizer.var.short.columns)

    # NOTE: weights are in 'chain' order, whereas 'var.short' is in the order returned from :meth:`.LogFrame.organize_time`
    weights = optimizer.weights.eval(session=optimizer.sess)[:, np.argsort(optimizer.chain)][:, short_idx]

    short = optimizer.var.short
    fn = short.columns.names.index('filename')
    height = short.shape[1]

    for i, s in enumerate(short.iteritems()):
        x = s[1].dropna()
        start_t = x.index[0]
        p = plt.plot(x)[0]

        idx = optimizer.var.index[weights[:, i].astype(bool)]
        ax.axvspan(idx.min(), idx.max(), alpha=.4, facecolor=p.get_color())

        ax.annotate(s[0][fn], xy=(start_t, 1 - (1 + i) / height),
                    xycoords=('data', 'axes fraction'), color=p.get_color())

    bx = ax.twinx()
    bx.plot(optimizer.var.index[1:], optimizer.resid.eval(session=optimizer.sess), color='grey', alpha=.5)
    fig.show()
