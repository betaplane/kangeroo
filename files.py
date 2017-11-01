import pandas as pd
import numpy as np
from glob import glob
import os, re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# from sklearn.tree import DecisionTreeClassifier


class Log(object):
    """Class which collects methods to read data logger (.csv) files. The class-level variables hold some information relevant to the parsing and processing of the files and can be adapted."""

    logger_columns = [
        ['Date', 'Time', 'LEVEL', 'TEMPERATURE'],
        ['Date', 'Time', 'Level', 'Temperature']
    ]
    """Column names in the data logger files. Each sub-list is tried until a match is found; if the names are different, an error will result."""

    column_names = ['variable', 'in_out', 'filename', 'id', 'offset', 'data_flag']
    """Levels in :class:`pandas.MultiIndex` constructed by this class' methods."""

    @staticmethod
    def _skip(filename):
        """Returns the number of lines to be skipped at the head of a data logger file."""
        with open(filename) as f:
            for i, line in enumerate(f):
                if re.search('date.*time', line, re.IGNORECASE):
                    return i

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
                d = pd.read_csv(filename, skiprows = cls._skip(filename), usecols=cols)
            except ValueError:
                pass
            else:
                break

        d.index = pd.DatetimeIndex(d.apply(lambda r:'{} {}'.format(r.Date, r.Time), 1))
        d.drop(['Date', 'Time'], 1, inplace=True)
        return cls._meta_multi_index(d, os.path.basename(os.path.splitext(filename)[0]))

    @classmethod
    def concat_directory(cls, directory):
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

    @classmethod
    def plot(cls, df, flags=None, right=0.8):
        x = df.xs('data', 1, 'data_flag')
        # flags = df.xs('flag', 1, 'data_flag') if flags is None else flags
        filenames = x.columns.names.index('filename')
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=right)
        used_or_not = {True:[], False:[]}
        for i, series in enumerate(df.groupby(axis=1, level='filename')):
            name = series[0]
            data = series[1].xs('data', 1, 'data_flag')
            flag = series[1].xs('flag', 1, 'data_flag')
            p = ax.plot(data)[0]
            u = False
            try:
                spans = cls.get_time_ranges(flags.xs(name, 1, 'filename'))
                print(name, spans)
                for span in spans:
                    ax.axvspan(*span, ymax=1-i/x.shape[1], ymin=1-(1+i)/x.shape[1], alpha=.4, facecolor=p.get_color())
                    u = True
            except:
                pass
            used_or_not[u].append(Patch(color = p.get_color(), label=name))

        ax.add_artist(plt.legend(handles = used_or_not[True], bbox_to_anchor=(1, 1), title='used'))
        plt.legend(handles = used_or_not[False], bbox_to_anchor=(1, 0), loc='lower left', title='not used')
        fig.show()

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
    def get_time_ranges(column):
        """Return a list of ranges (start, stop) of the data that have been used in a concatenation, as reconstructed from one columns of a DataFrame of (1, 0) flags. The flag DataFrame can be obtained by a call to :meth:`load_flags`. The arguments needs to be a proper :class:`~pandas.DataFrame` - see use in :meth:`plot`.

        """
        col = column.dropna()
        start, stop = (lambda idx: (idx.min(), idx.max()))(col)
        d = col.diff()
        # return d[d!=0].dropna().to_frame(name='a').sort_index().reset_index().pivot(columns='a', values='index')[[1, -1]]
        ranges = [[start]] if col.iloc[0] else []
        for t, s in d[d!=0].dropna().sort_index().iterrows():
            if np.all(s==1):
                ranges.append([t])
            else:
                ranges[-1].append(t)
        if len(ranges[-1])==1:
            ranges[-1].append(stop)
        return ranges

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

    def __init__(self, directory):
        self.data = self.concat_directory(directory)
        self.temp = self.get_variable(self.data, 'temp')
        self.level = self.get_variable(self.data, 'level')
