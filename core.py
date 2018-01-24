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


class Reader(object):
    """Base class which collects methods to read data logger (.csv) files.

    :Keyword arguments:
        * **directory** - The directory from which the data logger files are to be read.
        * **copy** - If ``copy=Reader`` where ``Reader`` is an instance of the :class:`Reader` class, the data attributes are simply copied over so that the don't need to be read again from the original files. This is for development and will be removed later.

    .. attribute:: data

        The :class:`~pandas.DataFrame` containing the read-in data with a :class:`~pandas.DatetimeIndex` and the columns corresponding to the files in ``directory``
    """

    logger_columns = [
        ['Date', 'Time', 'LEVEL', 'TEMPERATURE'],
        ['Date', 'Time', 'Level', 'Temperature'],
        {1: 'Date', 2: 'Time', 4: 'level', 5: 'temp'}
    ]
    """Column names in the data logger files. Each sub-list is tried until a match is found; if the names are different, an error will result."""

    def __init__(self, directory=None, copy=None):
        if directory is not None:
            files = glob(os.path.join(directory, '*.csv'))
            self.data = pd.concat([self.read(f) for f in files], 1)
            self.directory = directory
        else:
            self.data = copy.data
            self.directory = copy.directory

        self.temp = self.data.xs('temp', 1, 'var')
        self.level = self.data.xs('level', 1, 'var')

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
    def read(cls, filename):
        """Read a data logger .csv file and return a dictionary of DataFrames for individual columns of the file. Each DataFrame contains one column with the data and one column with a flag value (for subsequent use) which is set to 1 for each record. The :class:`~pandas.MultiIndex` has the levels:
            * *file* - the original filename (without extension) from which the data was read
            * *var* - the variable name (from the logger file)

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
            d.columns = [n.casefold() for n in d.columns]
            return pd.concat((d, ), 1, keys=[os.path.basename(os.path.splitext(filename)[0])],
                             names=['file', 'var'])
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
            file_names = S[variable].columns.get_level_values('file').dropna()
        return {'directory': set(files) - set(file_names), 'HDF5': set(file_names) - set(files)}
