import pandas as pd
import numpy as np
from warnings import warn
from glob import glob
import os, re
from .plot import Plots, get_time_ranges


class FileReadError(Exception):
    def __init__(self, msg, error):
        msg = """
        {} could not be read. Try saving this file with UTF-8 encoding.
        """.format(msg)
        super().__init__(msg, error)


# somehow sphinx doesn't see classes that inherit from 'mocked' imports - but the sphinx build failes if I don't mock pandas
# solution from here: https://stackoverflow.com/questions/29992444/sphinx-autodoc-skips-classes-inherited-from-mock
class LogFrame(pd.DataFrame):
    """:class:`pandas.DataFrame` subclass with less verbose accessor methods to some columns of the hierarchical :class:`~pandas.MultiIndex`. """

    @property
    def plot(self):
        return Plots(self)

    @property
    def _constructor(self):
        return LogFrame

    def _level(self, attr, level):
        try:
            return self.xs(attr, 1, level, False)
        except KeyError:
            warn("LogFrame does not have a '{}' attributes - original frame returned.".format(attr))
            return self

    @property
    def data(self):
        "Return `data` columns of `data_flag` level."
        return self._level('data', 'data_flag')

    @property
    def flag(self):
        "Return `flag` columns of `data_flag` level."
        return self._level('flag', 'data_flag')

    def variable(self, pattern):
        """Return all columns matching the given expression in the `variable` level of the :class:`~pandas.MultiIndex` of the parsed :class:`LogFrame`."""
        return self.loc[:, self.columns.get_level_values('variable').str.contains(pattern, case=False)]

    @property
    def end_points(self):
        def get_start_end(col):
            x = col.dropna()
            return pd.Series((x.index.min(), x.index.max()), index=['start', 'end'])

        return self.data.apply(get_start_end)

    @property
    def time_ranges(self):
        """Return a list of ranges (start, end) of the data that have been used in a concatenation, as reconstructed from one columns of a LogFrame of (1, 0) flags. The flag LogFrame can be obtained by a call to :meth:`load_flags`. The arguments needs to be a proper :class:`~pandas.DataFrame` / :class:`LogFrame` (as opposed to a :class:`~pandas.Series`) - see use in :meth:`plot`.

        """
        if self.flag.columns.get_level_values('data_flag').unique().item() == 'flag':
            return self.flag.apply(get_time_ranges)
        else:
            return self.notnull().astype(int).apply(get_time_ranges)


    def organize_time(self, length=100, indexers=False):
        """Split the LogFrame into two according to the length of the Series. Return either the column indexes for the long and short series, or a LogFrame with an added column level `length` with ``long`` and ``short`` labels.

        :param length: Threshold which divides long from short time series (in days).
        :param indexers: If ``True``, return only indexers, otherwise return LogFrame with added hierarchical column level.
        :type indexers: :obj:`bool`
        :returns: tuple of (long, short) indexers or LogFrame with top column level labels ``long`` and ``short``
        :rtype: :obj:`tuple` of :class:`numpy.ndarray` or :class:`LogFrame`

        """
        time = self.end_points.sort_values('start', 1)
        d = time.diff(1, 0).drop('start')
        l = d[d > pd.Timedelta(length, 'D')]
        long = l.dropna(1)
        short = d[l.isnull()].dropna(1)
        if indexers:
            return self.columns.get_indexer(long.columns), self.columns.get_indexer(short.columns)
        else:
            df = pd.concat((self[long.columns], self[short.columns]), 1, keys=['long', 'short'])
            names = self.columns.names.copy()
            names.insert(0, 'long_short')
            df.columns.names = names
            return df


class Reader(object):
    """Base class which collects methods to read data logger (.csv) files.

    :Keyword arguments:
        * **directory** - The directory from which the data logger files are to be read.
        * **copy** - If ``copy=Reader`` where ``Reader`` is an instance of the :class:`Reader` class, the data attributes are simply copied over so that the don't need to be read again from the original files. This is for development and will be removed later.
        * **column_names** - if passed, replaces the class-level :attr:`.column_names`.
        * Any further arguments are passed through to :meth:`_meta_multi_index` - they can be used to add additional levels to the columns :obj:`~pandas.MultiIndex` or to remove default ones by setting them to ``None``, e.g. ``data_flag = None``.

    """

    logger_columns = [
        ['Date', 'Time', 'LEVEL', 'TEMPERATURE'],
        ['Date', 'Time', 'Level', 'Temperature'],
        {1: 'Date', 2: 'Time', 4: 'level', 5: 'temp'}
    ]
    """Column names in the data logger files. Each sub-list is tried until a match is found; if the names are different, an error will result."""

    column_names = ['variable', 'in_out', 'filename', 'id', 'offset', 'data_flag']
    """Levels in :class:`pandas.MultiIndex` constructed by this class' methods."""

    column_defaults = {'data_flag': 'data', 'in_out': 'in', 'id': 0., 'offset': 0., 'variable': ''}

    def __init__(self, directory=None, copy=None, **kwargs):
        if directory is not None:
            files = glob(os.path.join(directory, '*.csv'))
            self.data = LogFrame(pd.concat([self.read(f, **kwargs) for f in files], 1))
        else:
            self.data = copy.data

        self.temp = self.data.variable('temp')
        self.level = self.data.variable('level')

    @staticmethod
    def _skip(filename):
        """Returns the number of lines to be skipped at the head of a data logger file"""
        try:
            with open(filename) as f:
                for i, line in enumerate(f):
                    if re.search('date.*time', line, re.IGNORECASE):
                        return i
            return 0
        except Exception as ex:
            raise FileReadError(filename)

    @classmethod
    def _meta_multi_index(cls, df, columns, **kwargs):
        """Add metadata in the form of :class:`~pandas.MultiIndex` columns to a DataFrame read in by a call to :func:`~pandas.read_csv`. Used by :meth:`read`.

        :param df: Input DataFrame
        :type df: :class:`~pandas.DataFrame`
        :param filename: the name of the csv data logger file
        :returns: DataFrame with metadata in columns' :class:`~pandas.MultiIndex` - see :meth:`read` for a description of the index levels.
        :rtype: :class:`~pandas.DataFrame`

        :Keyword arguments:
            See the main :class:`Reader` docstring.

        """

        column_names = kwargs.pop('column_names', cls.column_names)
        defaults = cls.column_defaults.copy()
        defaults.update(kwargs)
        column_names = [c for c in column_names if defaults[c] is not None]
        column_names.extend([k for k, v in kwargs.items() if (v is not None) and (k not in column_names)])

        # turn around the order
        for level in column_names[::-1]:
            if level != columns:
                df = pd.concat((df, ), 1, keys=[defaults[level]])

        a = len(df.columns.levels) - 1
        b = column_names.index(columns)
        for i in range(a, b, -1):
            df.columns = df.columns.swaplevel(i, i-1)

        df.columns.names = column_names

        if 'data_flag' in column_names:
            flags = df.copy()
            flags.loc[:, :] = 1
            flags.columns = flags.columns.set_levels(['flag'], 'data_flag')
            df = pd.concat((df, flags), 1).sort_index(1)
        return df

    @classmethod
    def read(cls, filename, **kwargs):
        """Read a data logger .csv file and return a dictionary of DataFrames for individual columns of the file. Each DataFrame contains one column with the data and one column with a flag value (for subsequent use) which is set to 1 for each record. The :class:`~pandas.MultiIndex` has the levels:
            * *variable* - the variable name (from the logger file)
            * *in_out* - whether this is raw data ('in') or concshortatenated ('out')
            * *filename* - the original filename (without extension) from which the data was read
            * *id* - probably just the id of the field in the old 'databarc' database, 0. for newer data
            * *offset* - any additive offset applied to the data when used in a concatenation (the data in the columns is left unchanged)
            * *data_flag* - whether this is a data or a flag column

        The columns which are read are given in the :attr:`logger_columns` class variable.

        :param filename: csv file to be read
        :returns: DataFrame with metadata in the columns :class:`~pandas.MultiIndex` and a :class:`~pandas.DatetimeIndex` as index. The timestamps are constructed from the columns 'Date' and 'Time' in the datalogger files.
        :rtype: :class:`~pandas.DataFrame`

        :Keyword arguments:
            Are passed directly to :meth:`._meta_multi_index`.

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
            d.columns = [n.casefold() for n in d.columns]
            return cls._meta_multi_index(d, filename=os.path.basename(os.path.splitext(filename)[0]), columns='variable', **kwargs)
        except UnboundLocalError:
            raise Exception('problems with {}'.format(filename))


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
