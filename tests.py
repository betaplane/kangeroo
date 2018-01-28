from unittest import TestCase
from .concat import Concatenator
import tarfile, os, shutil
import pandas as pd
import numpy as np

class LevelTest(TestCase):
    @classmethod
    def setUpClass(cls):
        tf = tarfile.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_level_4_1.tar.gz'))
        tf.extractall('test_dir')
        variable = 'level'
        resample = '30T'
        cls.out = pd.read_csv(os.path.join('test_dir', 'out', '{}_output.csv'.format(variable)),
                                  parse_dates=True, index_col=0).resample(resample).asfreq()
        shutil.rmtree('test_dir/out')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('test_dir')


    # test are run in lexigraphically sorted order, so this is # 1
    def test_all_files_level(self):
        cc = Concatenator(directory='test_dir', variable='level', correct_time=True)
        cc.concat(no_offset=[5,17], use_spline=[3])
        np.testing.assert_allclose(self.out['concat'].dropna(), cc.out['concat'].dropna())
        cc.var.drop(cc.var.columns[-1], 1, inplace=True)
        cc.out = pd.DataFrame(np.nan, index=cc.var.index, columns=['resid', 'extra', 'interp', 'outliers', 'concat'])
        cc.traverse()
        cc.concat()
        cc.concat(no_offset=[5,17], use_spline=[3])
        cc.to_csv()

    # 2
    def test_append_file_level(self):
        cc = Concatenator(directory='test_dir', variable='level')
        cc.concat(no_offset=[0])
        cc.to_csv() # important - the arrays are only the same after applying to_csv()
        np.testing.assert_allclose(self.out['concat'].dropna(), cc.out['concat'].dropna())

if __name__ == '__main__':
    unittest.main()
