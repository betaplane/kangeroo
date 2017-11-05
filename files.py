"""
Usage Example
=============
.. code-block::

    log = Log('data/4/1/')
    long, short = Log.organize_time(log.level)
    Log.plot(long)

    c = Log.list(short)
    Log.plot(c[3], cut_ends=True)

"""
import pandas as pd
import numpy as np
from glob import glob
import os, re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from types import MethodType
from statsmodels.tsa import ar_model
import sklearn.cluster as cluster
import time

class DataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return DataFrame

    def _data_flag(self, attr, level):
        try:
            return self.xs(attr, 1, level)
        except KeyError:
            return self

    @property
    def data(self):
        return self._data_flag('data', 'data_flag')

    @property
    def flag(self):
        return self._data_flag('flag', 'data_flag')


class FileReadError(Exception):
    def __init__(self, msg, error):
        msg = """
        {} could not be read. Try saving this file with UTF-8 encoding.
        """.format(msg)
        super().__init__(msg, error)

class Log(object):
    """Class which collects methods to read data logger (.csv) files. The class-level variables hold some information relevant to the parsing and processing of the files and can be adapted."""

    logger_columns = [
        ['Date', 'Time', 'LEVEL', 'TEMPERATURE'],
        ['Date', 'Time', 'Level', 'Temperature'],
        {1: 'Date', 2: 'Time', 4: 'level', 5: 'temp'}
    ]
    """Column names in the data logger files. Each sub-list is tried until a match is found; if the names are different, an error will result."""

    column_names = ['variable', 'in_out', 'filename', 'id', 'offset', 'data_flag']
    """Levels in :class:`pandas.MultiIndex` constructed by this class' methods."""

    @staticmethod
    def _skip(filename):
        """Returns the number of lines to be skipped at the head of a data logger file."""
        try:
            with open(filename) as f:
                for i, line in enumerate(f):
                    if re.search('date.*time', line, re.IGNORECASE):
                        return i
            return 0
        except Exception as ex:
            raise FileReadError(filename)

    @classmethod
    def _meta_multi_index(cls, df, name=None):
        """Add metadata in the form of :class:`~pandas.MultiIndex` columns to a DataFrame read in by a call to :func:`~pandas.read_csv`. Used by :meth:`read`.

        :param series: Input DataFrame
        :type series: :class:`~pandas.DataFrame`
        :param name: the name of the csv data logger file
        :returns: DataFrame with metadata in columns' :class:`~pandas.MultiIndex` - see :meth:`read` for a description of the index levels.
        :rtype: :class:`~pandas.DataFrame`

        """
        defaults = {'data_flag': 'data', 'in_out': 'in', 'filename': name, 'id': 0., 'offset': 0.}

        # turn around the order and leave out the last level because it needs to be swapped with the 'variable' level eventually
        for level in cls.column_names[:-1][::-1]:
            df = pd.concat((df, ), 1, keys=[defaults[level]]) if level in defaults else df
        # here I add the last level so I can then swap it
        df = pd.concat((df, ), 1, keys=[defaults[cls.column_names[-1]]])

        df = df.swaplevel(0, -1, 1)
        df.columns.names = cls.column_names

        # add the flag columns
        flags = df.copy()
        flags.loc[:, :] = 1
        flags.columns = flags.columns.set_levels(['flag'], 'data_flag')
        return pd.concat((df, flags), 1).sort_index(1)

    @classmethod
    def read(cls, filename):
        """Read a data logger .csv file and return a dictionary of DataFrames for individual columns of the file. Each DataFrame contains one column with the data and one column with a flag value (for subsequent use) which is set to 1 for each record. The :class:`~pandas.MultiIndex` has the levels:
            * *variable* - the variable name (from the logger file)
            * *in_out* - whether this is raw data ('in') or concatenated ('out')
            * *filename* - the original filename (without extension) from which the data was read
            * *id* - probably just the id of the field in the old 'databarc' database, 0. for newer data
            * *offset* - any additive offset applied to the data when used in a concatenation (the data in the columns is left unchanged)
            * *data_flag* - whether this is a data or a flag column

        The columns which are read are given in the :attr:`logger_columns` class variable.

        :param filename: csv file to be read
        :returns: DataFrame with metadata in the columns :class:`~pandas.MultiIndex` and a :class:`~pandas.DatetimeIndex` as index. The timestamps are constructed from the columns 'Date' and 'Time' in the datalogger files.
        :rtype: :class:`~pandas.DataFrame`

        """
        print("Reading file {}".format(filename))

        # provide some alternatives for different logger file formats
        for cols in cls.logger_columns:
            try:
                if isinstance(cols, dict):
                    cols, names = zip(*cols.items())
                else:
                    names = None
                d = pd.read_csv(filename, skiprows = cls._skip(filename), usecols=cols, names=names)
            except ValueError as err:
                pass
            except:
                raise
            else:
                break

        try:
            d.index = pd.DatetimeIndex(d.apply(lambda r:'{} {}'.format(r.Date, r.Time), 1))
            d.drop(['Date', 'Time'], 1, inplace=True)
            return cls._meta_multi_index(d, os.path.basename(os.path.splitext(filename)[0]))
        except UnboundLocalError:
            raise Exception('problems with {}'.format(filename))

    @classmethod
    def read_directory(cls, directory):
        files = glob(os.path.join(directory, '*.csv'))
        return pd.concat([cls.read(f) for f in files], 1)

    @staticmethod
    def get_variable(df, pattern):
        """Return all columns matching the given expression in the `variabls` level of the :class:`~pandas.MultiIndex` of the parsed :class:`~pandas.DataFrame`."""
        return df.loc[:, df.columns.get_level_values('variable').str.contains(pattern, case=False)]

    @staticmethod
    def get_start_stop(col): # apply to axis 0
        x = col.dropna()
        return pd.Series((x.index.min(), x.index.max()), index=['start', 'stop'])

    @classmethod
    def organize_time(cls, df, length=100):
        """Return a tuple of two DataFrames which partition the input DataFrame into long and short time series, as separated by a length thresold.

        :param df: DataFrame of raw record time series
        :type df: :class:`~pandas.DataFrame`
        :param length: length threshold in days which separates long from short time series
        :returns: tuple of DataFrames of (long, short) time series
        :rtype: :obj:`tuple` of :class:`DataFrames<pandas.DataFrame>`

        """
        time = df.apply(cls.get_start_stop, 0).sort_values('start', 1)
        d = time.diff(1, 0).drop('start')
        l = d[d > pd.Timedelta(length, 'D')]
        long = l.dropna(1)
        short = d[l.isnull()].dropna(1)
        return df[long.columns], df[short.columns]

    @staticmethod
    def get_time_ranges(column):
        """Return a list of ranges (start, stop) of the data that have been used in a concatenation, as reconstructed from one columns of a DataFrame of (1, 0) flags. The flag DataFrame can be obtained by a call to :meth:`load_flags`. The arguments needs to be a proper :class:`~pandas.DataFrame` - see use in :meth:`plot`.

        """
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

    @classmethod
    def plot(cls, df, flags=None, right=0.8, flagged=True, cut_ends=False):
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=right)

        flags = df.xs('flag', 1, 'data_flag') if flags is None else flags

        # need to sort the series first by start time
        flags = flags.apply(cls.get_time_ranges)
        names = flags.loc[0].apply(lambda c:c[0]).sort_values().index.get_level_values('filename')
        height = len(names)

        used_or_not = {True:[], False:[]}
        for i, name in enumerate(names):
            d = df.xs(name, 1, 'filename')
            # p = ax.plot(d.xs('data', 1, 'data_flag'))[0]
            p = cls.plot_flagged(d, ax, cut_ends=cut_ends, residuals=False)
            u = False
            try:
                spans = flags.xs(name, 1, 'filename')
                for _, span in spans.iterrows():
                    ax.axvspan(*span.item(), ymax=1-i/height, ymin=1-(1+i)/height,
                               alpha=.4, facecolor=p.get_color())
                    u = True
            except:
                pass
            used_or_not[u].append(Patch(color = p.get_color(), label=name))

        ax.add_artist(
            plt.legend(handles = used_or_not[False], loc='lower left', bbox_to_anchor=(1, 0), title='not used'))
        plt.legend(handles = used_or_not[True], bbox_to_anchor=(1, 1), title='used')

        fig.show()

    @classmethod
    def cut_ends(cls, df, std=3, lag=1, residuals=False):
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

    @classmethod
    def plot_flagged(cls, df, ax=None, cut_ends=False, residuals=True, **kwargs):
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [None]

        df = df.dropna(0, 'all')
        if cut_ends:
            df, resid = cls.cut_ends(df, residuals=True, **kwargs)
        data = df.xs('data', 1, 'data_flag')
        flags = df.xs('flag', 1, 'data_flag')

        (fig, ax) = plt.subplots() if ax is None else (ax.figure, ax)
        p = ax.plot(data, color=colors[0])
        ax.plot(data[(flags.isnull()) | (flags==0)], 'or')
        try:
            ax.plot(data.loc[[i for j in cls.get_time_ranges(flags) for i in j]], 'og')
        except:
            pass
        if residuals:
            bx = ax.twinx()
            bx.plot(resid, color=colors[1])
        if ax is None:
            fig.show()
        else:
            return p[0]

    @staticmethod
    def check_directory(filename, variable, base_path='.'):
        """Check the contents of a logger file directory against the filenames in the `filename` level of the :class:`~pandas.MultiIndex` columns of a processed dataframe.

        :param filename: name of the :class:`~pandas.HDFStore` file containing the processed DataFrames
        :param variable: name of the variable within that file (full 'path')
        :param base_path: base path where the corresponding directory tree is rooted
        :returns: a dictionary of lists which contain the filenames present exclusively in one of the two repositories - the :class:`~pandas.HDFStore` ('HFD5') and the directory ('directory')
        :rtype: :obj:`dict` of lists

        """
        files = os.listdir(os.path.join(base_path, os.path.split(variable)[0]))
        files = [os.path.splitext(f)[0] for f in files]
        with pd.HDFStore(filename) as S:
            file_names = S[variable].columns.get_level_values('filename').dropna()
        return {'directory': set(files) - set(file_names), 'HDF5': set(file_names) - set(files)}

    @staticmethod
    def load_flags(filename, variable):
        """Load a DataFrame with flag values corresponding to a given concatenation.

        :param filename: name of the :class:`~pandas.HDFStore` file to be used
        :param variable: name of the variable within the file
        :returns: a DataFrame of flag values from which a concatenation can be constructed
        :rtype: :class:`~pandas.DataFrame`

        """
        idx = pd.IndexSlice
        with pd.HDFStore(filename) as S:
            flags = S[variable].sort_index(1).loc[:, idx['in', :, :, :, 'flag']]
        return flags
        # t = np.array(flags.index, dtype='datetime64', ndmin=2).astype(float).T

    @staticmethod
    def process(df, from_time = None, merge = True):
        x = df if from_time is None else df.loc[from_time:]
        out = x.xs('out', 1, 'in_out', False)
        y = x.xs('in', 1, 'in_out')
        d = y.xs('data', 1, 'data_flag')
        z = (d * y.xs('flag', 1, 'data_flag'))
        z = z.add(d.columns.get_level_values('offset').astype(float), 1).mean(1)
        f = out.xs('flag', 1, 'data_flag').iloc[:,0]
        z[f==-1] = np.nan
        z = z.interpolate('time')
        if merge is False:
            return z
        z = z.to_frame()
        z.columns = pd.MultiIndex.from_tuples(out.xs('data', 1, 'data_flag', False).columns)
        z.columns.names = df.columns.names
        return z.combine_first(df)

    @classmethod
    def list(cls, df):
        """Return list of DataFrames, each corresponding to a single file name and having the `data` and `flag` subcolumns. The list is ordered according to the start times of the **used** values (as determined by a `flag` value of 1.)."""
        flags = df.xs('flag', 1, 'data_flag').apply(cls.get_time_ranges)
        idx = flags.loc[0].apply(lambda c:c[0]).sort_values().index.get_level_values('filename')
        g = df.groupby(axis=1, level='filename')
        return [g.get_group(f) for f in idx]

    def __init__(self, directory):
        self.data = DataFrame(self.read_directory(directory))
        self.temp = DataFrame(self.get_variable(self.data, 'temp'))
        self.level = DataFrame(self.get_variable(self.data, 'level'))


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

    @classmethod
    def chains(cls, df):
        """Attempt at a method that finds all permutations of combinations based on overlaps."""
        s = df.apply(cls.get_start_stop)
        overl, cont = cls.overlap(s)
        idx = df.columns.get_indexer(s.loc['start'].sort_values().index)

        def stack(i):
            n = list(set(np.where(overl[i[-1]])[0]) - set(i))
            print(i, n)
            if len(n) == 0:
                j = np.where(idx == i[-1])[0]
                n = idx[j + 1]
            return np.hstack((i.reshape((1, -1)).repeat(len(n), 0), np.array(n).reshape((-1, 1))))

        c = idx[0].reshape((1, 1))
        for _ in range(len(idx)):
            try:
                c = np.vstack([stack(i) for i in c])
            except:
                break
        return c


if __name__ == '__main__':
    # l = Log('2/2/AT')
    pass
