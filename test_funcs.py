# ----------------------------------------------------------------------------
# Created By: Widner Martin (martin.widner@iu-study.org)
# Matriculation number: 321147833
# Created Date: 20/10/2021
# version ='1.0''
# ---------------------------------------------------------------------------
"""
The purpose of this module is to run unittests from various methods defined in classes_functions.py
The tests focus primarily on the methods of the data objects, since they form the basis of this project
"""

import classes_functions as cf
import unittest


class TestFuncs(unittest.TestCase):

    def setUp(self):
        """
        setUp method to be executed before each test, since the train, test and ideal data object
        is used for more than one test method in unittest
        """
        self.test_data = cf.DataNotIdeal("test.csv")
        self.train_data = cf.DataNotIdeal("train.csv")
        self.ideal_data = cf.Data('ideal.csv')

    def test_exception_import(self):
        """
        test if an Exception is raised, when the import data has not the correct format (.txt instead of .csv)
        """
        with self.assertRaises(Exception):
            cf.Data("ideal.txt")

    def test_return_xy_data(self):
        """
        test if x-y-data is returned correctly and if an index error occurs
        """
        # test of correctly returned values according to provided .csv-files
        self.assertEqual(self.test_data.return_xy_data(4, 1), 66.35596)
        self.assertEqual(self.train_data.return_xy_data(11, 3), 2.3855636)
        self.assertEqual(self.ideal_data.return_xy_data(40, 5), -10.287904)

        # test of index error occurs
        with self.assertRaises(IndexError):
            self.test_data.return_xy_data(3, 10)
        with self.assertRaises(IndexError):
            self.train_data.return_xy_data(100, 20)
        with self.assertRaises(IndexError):
            self.ideal_data.return_xy_data(50, 67)

    def test_return_y_data(self):
        """
        test if index error occurs
        """
        with self.assertRaises(IndexError):
            self.test_data.return_y_data(7)
        with self.assertRaises(IndexError):
            self.train_data.return_y_data(8)
        with self.assertRaises(IndexError):
            self.ideal_data.return_y_data(60)


if __name__ == '__main__':
    unittest.main()
