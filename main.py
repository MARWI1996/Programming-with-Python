# ---------------------------------------------------------------------------
# Created By: Widner Martin (martin.widner@iu-study.org)
# Matriculation number: 321147833
# Created Date: 14/10/2021
# version ='1.0'
# ---------------------------------------------------------------------------
"""
This module is the main.py-file for the assignment "Programming with Python (DLMDSPWP01)".
The tasks are according to the given task description. The main tasks are summarized as:
- loading given data sets from .csv-files (train, test, ideal data)
- finding ideal functions with the help of train data, by minimizing the sum of least squares
- mapping each test data point individually to the determined ideal function, with respect to the mapping criterion
- saving data in a SQL database
- visualization of the results
The defined classes and functions are imported from the classes_functions.py file
"""

import classes_functions as cf


def main():
    # Creation of the data objects
    # test data
    test_data = cf.DataNotIdeal("test.csv")
    # train data
    train_data = cf.DataNotIdeal("train.csv")
    # ideal data
    ideal_data = cf.Data("ideal.csv")

    # Find the matching ideal functions
    # least squares method or least absolute deviations (LAD) method, select in classes_functions.py file
    cf.find_ideal_func(train_data, ideal_data)
    # if Huber loss method is used, comment out the line above and activate the line below
    # cf.find_ideal_func_huber(train_data, ideal_data)
    ideal_funcs_indices = train_data.return_ideal_func_index()
    ideal_funcs_errors = train_data.return_ideal_func_max_error()

    # Test point mapping: Determine the mapping and the smallest errors
    (mapping, mapped_smallest_error) = cf.test_mapping(test_data, ideal_data, ideal_funcs_indices, ideal_funcs_errors)

    # Creation of SQL database and filling in the data
    cf.create_database(train_data, test_data, ideal_data, mapping, mapped_smallest_error, ideal_funcs_indices)

    # Extracting the relevant data for visualization
    # Extraction of the test data points assigned to the determined ideal function 1
    (x_values1, y_values1) = cf.extract_data(mapping, test_data, 1)
    # Extraction of the test data points assigned to the determined ideal function 2
    (x_values2, y_values2) = cf.extract_data(mapping, test_data, 2)
    # Extraction of the test data points assigned to the determined ideal function 3
    (x_values3, y_values3) = cf.extract_data(mapping, test_data, 3)
    # Extraction of the test data points assigned to the determined ideal function 4
    (x_values4, y_values4) = cf.extract_data(mapping, test_data, 4)
    # Extraction of the test data points not assigned to any ideal function
    (x_values_none, y_values_none) = cf.extract_data(mapping, test_data, "None")

    # Plot of the results
    # Determined ideal functions with minimized sum of least squares
    cf.plot_ideal_functions(train_data, ideal_data, ideal_funcs_indices)

    # Results of the mapping
    # Plot of the mappings for the first determined ideal function
    cf.plot_mapping(ideal_data, ideal_funcs_errors, ideal_funcs_indices, x_values1, y_values1, 1)
    # Plot of the mappings for the second determined ideal function
    cf.plot_mapping(ideal_data, ideal_funcs_errors, ideal_funcs_indices, x_values2, y_values2, 2)
    # Plot of the mappings for the third determined ideal function
    cf.plot_mapping(ideal_data, ideal_funcs_errors, ideal_funcs_indices, x_values3, y_values3, 3)
    # Plot of the mappings for the fourth determined ideal function
    cf.plot_mapping(ideal_data, ideal_funcs_errors, ideal_funcs_indices, x_values4, y_values4, 4)

    # Plot of the mapping distribution
    cf.plot_distribution(len(x_values1), len(x_values2), len(x_values3), len(x_values4), len(x_values_none))


if __name__ == '__main__':
    main()
