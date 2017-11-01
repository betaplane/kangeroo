"""
Remember
========
* :meth:`DB.get_fields` now should work with `mult` factor from `fields` table

"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

class DB(object):
    """
    Class for interacting with the existing 'databarc' database. All tables can be loaded by accessing an attribute of the same name as the table (e.g. `DB.fields`).

    """

    def __init__(self, uri='postgresql://arno@/AKR'):
        from sqlalchemy import create_engine
        self.engine = create_engine(uri)
        """:class:`sqlalchemy.engine.Engine` connectable"""

    def get_fields(self, *ids):
        """Return a DataFrame with data (column `x`) and flags (column `info`) from the field(s) with the given ids.

        :Positional Arguments:
            * **\*ids** - Argument list of field ids (**not** station_ids). Field ids can be found from the :attr:`field` table/attribute.
        :returns: DataFrame with :class:`~pandas.MultiIndex` containing the field ids and `x` and `info` (i.e. data and flag) columns
        :rtype: :class:`pandas.DataFrame`

        """
        df = pd.read_sql(
            text("""
            SELECT
            r.id, r.t, r.field_id, r.info, ri.x as xint, rf.x as xfloat, rn.x as xnum, f.mult as mult
            FROM
            record r LEFT OUTER JOIN record_int ri ON r.id = ri.id
            LEFT OUTER JOIN record_float rf ON r.id = rf.id
            LEFT OUTER JOIN record_num rn ON r.id = rn.id
            LEFT OUTER JOIN field f ON r.field_id = f.id
            WHERE field_id=ANY(:ids);
            """), self.engine, parse_dates='t', index_col='id',
            params = {'ids': list(ids)}).dropna(1, 'all').sort_index()
        i = [k for k in ['xint', 'xfloat', 'xnum'] if k in df]
        x = df[i].sum(1) * df.mult
        x.name = 'x'
        return pd.concat((df.drop(i, 1).drop('mult', 1), x), 1).pivot('t', 'field_id')

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            if not self.engine.has_table(name):
                raise Exception('table does not exist in database')
            table = pd.read_sql_table(name, self.engine, index_col='id')
            setattr(self, name, table)
            return table

    def concat(self, out_id):
        """Reconstruct a concatenation saved in the databarc database.

        :param out_id: The `field_id` of the (concatenated / 'output') field whose concatenation should be reconstructed
        :returns: A concatenation DataFrame of the format as constructed by the :mod:`pandas`-based part of this module (:class:`~darc.files.Log`)
        :rtype: :class:`~pandas.DataFrame` with :class:`~pandas.MultiIndex` of the same characteristics as those used by :class:`~darc.files.Log`

        """
        p = self.processing[(self.processing.output_id == out_id) & self.processing.use]
        offset = {v.input_id: v.offset for k, v in p.iterrows()}
        source = {k: v.source for k, v in self.field.iterrows()}
        ids = p.input_id.tolist()
        recs = self.get_fields(out_id, *ids)
        data = recs['x'].drop(out_id, 1)
        info = recs['info'].drop(out_id, 1)
        info = (((info == 0).astype(float) + info.isnull()) > 0).astype(float).replace(False, np.nan)
        info[data.isnull()] = np.nan
        info.columns = pd.MultiIndex.from_tuples([(source[j], j, offset[j], 'in') for j in ids])
        data.columns = info.columns
        x = pd.concat((data, info), 1, keys=['data', 'flag'])
        x.columns.names = ['data_flag', 'filename', 'id', 'offset', 'in_out']
        x = x.swaplevel(i = 'data_flag', j = 'in_out', axis=1)
        out = pd.concat((recs['x'][out_id], recs['info'][out_id]), 1)
        out.columns = pd.MultiIndex.from_tuples([
            ('out', None, out_id, 0., 'data'), ('out', None, out_id, 0., 'flag')])
        out.columns.names = x.columns.names
        return pd.concat((out, x), 1).reindex(recs['x'][out_id].dropna().index)
