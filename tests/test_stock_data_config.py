import os
import unittest
from unittest.mock import patch

from alphacfg.data.stock_data import StockData


class StockDataConfigTests(unittest.TestCase):
    def tearDown(self):
        StockData._region = None

    def test_region_can_be_selected_by_environment(self):
        with patch.dict(os.environ, {"ALPHACFG_QLIB_REGION": "us"}, clear=False):
            self.assertEqual(StockData._resolve_region("/tmp/cn_data"), "us")

    def test_us_data_directory_selects_us_region(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(StockData._resolve_region("/tmp/us_data"), "us")

    def test_invalid_region_is_rejected(self):
        with patch.dict(os.environ, {"ALPHACFG_QLIB_REGION": "eu"}, clear=False):
            with self.assertRaisesRegex(ValueError, "must be 'cn' or 'us'"):
                StockData._resolve_region("/tmp/data")


if __name__ == "__main__":
    unittest.main()
