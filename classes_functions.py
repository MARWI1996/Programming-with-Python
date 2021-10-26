# ----------------------------------------------------------------------------
# Created By: Widner Martin (martin.widner@iu-study.org)
# Matriculation number: 321147833
# Created Date: 14/10/2021
# version ='1.0''
# ---------------------------------------------------------------------------
"""
In this module the classes and functions are defined, which are needed for the assignment in the main.py file
"""

import numpy as np
import sqlalchemy as db
import os
import matplotlib.pyplot as plt
import statistics as stat


class Data(object):
    """
    Data object which is handling the provided data (in .csv-format)
    Names of the data sets need to be: test data "test.csv", train data "train.csv", ideal data "ideal.csv"
    """

    def __init__(self, filename):
        """
        class constructor: the data is loaded from the given .csv-files
        """
        try:
            self.f = open(filename, "rb")
            self.data = np.loadtxt(self.f, delimiter=",", skiprows=1)
            self.x_data = self.data[:, 0]
            self.num_rows, self.num_cols = self.data.shape
            self.f.close()
        except:
            raise Exception("Incorrect format of import file")

    def return_x_data(self):
        """
        :return: returns the x data of the data object
        """
        return self.x_data

    def return_y_data(self, i=1):
        """
        :param i: defines the index of the row which is returned
        :return: returns the y data of the data object
        """
        try:
            return self.data[:, int(i)]
        except:
            raise IndexError(f"Wrong input, only integers from 1 to {self.num_cols - 1} are allowed")

    def return_num_rows(self):
        """
        :return: returns the number of rows of the data object
        """
        return self.num_rows

    def return_num_cols(self):
        """
        :return: returns the number of columns of the data object
        """
        return self.num_cols

    def return_xy_data(self, row, column):
        """
        :param row: defines the index of the row which is returned
        :param column: defines the index of the column which is returned
        :return: returns one specific value of the data set, by defining the row and column
        """
        try:
            return self.data[row, column]
        except:
            raise IndexError("Given row and column are not within the range of the data object")


class DataNotIdeal(Data):
    """
    Inherited Data object for handling the not ideal data, with more specialized methods
    """

    def __init__(self, filename):
        """
        class constructor of the inheritance
        """
        super().__init__(filename)
        self.assigned_ideal_func = []
        self.sum_error = []
        self.max_error = []

    def assign_ideal_func(self, index_ideal_func, sum_error, max_error):
        """
        assign the found ideal function (index, sum error and max error) to the data object
        :param index_ideal_func: assigns the index of the determined ideal function
        :param sum_error: assigns the minimized sum of errors
        :param max_error: assigns the determined max error between determined ideal function and test data
        """
        self.assigned_ideal_func.append(index_ideal_func)
        self.sum_error.append(sum_error)
        self.max_error.append(max_error)

    def return_ideal_func_index(self):
        """
        :return: returns the assigned ideal functions
        """
        return self.assigned_ideal_func

    def return_ideal_func_error_sum(self):
        """
        :return: returns the determined minimized error sum
        """
        return self.sum_error

    def return_ideal_func_max_error(self):
        """
        :return: return the determined max errors for the mapping criterion
        """
        return self.max_error


def find_ideal_func(train_data, ideal_data):
    """
    Finding the ideal functions with the use of the train data by minimizing the sum of errors
    Calculation of the max. acceptable error for the mapping criterion and assigning to the train data
    As criterion for finding the ideal function, it can be switched between minimizing the least square
    deviations or minimizing the least absolute deviations (LAD) via removing and adding "#" in the correct line
    :param train_data: DataNotIdeal object "train_data"
    :param ideal_data: Data object "ideal_data"
    """
    # initialization of variables
    total_error_temp = 0.0
    total_error = None
    max_error = 0
    ideal_index = None

    # each column of train data
    for num in range(1, train_data.return_num_cols()):
        # each column of ideal data
        for i_columns in range(1, ideal_data.return_num_cols()):
            # compare each entry between ideal and train function
            for i_rows in range(0, ideal_data.return_num_rows()):
                # Least square deviation
                total_error_temp = total_error_temp + (
                        ideal_data.return_xy_data(i_rows, i_columns) - train_data.return_xy_data(i_rows, num)) ** 2
                # Least absolute deviations (LAD)
                # total_error_temp = total_error_temp + abs(
                #    ideal_data.return_xy_data(i_rows, i_columns) - train_data.return_xy_data(i_rows, num))

            if (total_error is None) or (total_error_temp <= total_error):
                total_error = total_error_temp
                ideal_index = i_columns
            else:
                pass

            # reset the temporary total error after comparing the train function with one ideal function
            total_error_temp = 0

        # determine max error between each train function and determined ideal function
        for m in range(0, ideal_data.num_rows):
            max_error_temp = ideal_data.return_xy_data(m, ideal_index) - train_data.return_xy_data(m, num)

            if (max_error is None) or (max_error_temp > max_error):
                max_error = max_error_temp
            else:
                pass

        # assign the determined index, minimized error and max. deviation from ideal function to the train object
        train_data.assign_ideal_func(ideal_index, total_error, max_error)

        # reset of variables after each train data column
        total_error = None
        max_error = 0
        ideal_index = None


def find_ideal_func_huber(train_data, ideal_data):
    """
    Finding the ideal functions with the use of the train data by minimizing the sum of errors
    Calculation of the max. acceptable error for the mapping criterion and assigning to the train data
    As criterion for finding the ideal function the Huber loss method is used
    :param train_data: DataNotIdeal object "train_data"
    :param ideal_data: Data object "ideal_data"
    """
    # initialization of variables
    total_error_temp = 0.0
    total_error = None
    max_error = 0
    ideal_index = None
    errors = []

    # each column of train data
    for num in range(1, train_data.return_num_cols()):
        # each column of ideal data
        for i_columns in range(1, ideal_data.return_num_cols()):
            # calculate all the absolute deviations between train and ideal function and add them to the error list
            for i_rows in range(0, ideal_data.return_num_rows()):
                e = abs(ideal_data.return_xy_data(i_rows, i_columns) - train_data.return_xy_data(i_rows, num))
                errors.append(e)

            # calculation of the median of the absolute deviations (MAD)
            mad = stat.median(errors)
            sigma = mad * 1.483
            k = sigma * 1.5

            # applying the Huber loss method
            for i_rows in range(0, ideal_data.return_num_rows()):
                error = ideal_data.return_xy_data(i_rows, i_columns) - train_data.return_xy_data(i_rows, num)

                if error <= abs(k):
                    total_error_temp = total_error_temp + error ** 2
                else:
                    total_error_temp = total_error_temp + (2 * k * abs(error) - k ** 2)

            if (total_error is None) or (abs(total_error_temp) <= total_error):
                total_error = abs(total_error_temp)
                ideal_index = i_columns
            else:
                pass

            # reset the temporary total error after comparing the train function with one ideal function
            total_error_temp = 0
            errors = []

        # determine max error between each train function and determined ideal function
        for m in range(0, ideal_data.num_rows):
            max_error_temp = ideal_data.return_xy_data(m, ideal_index) - train_data.return_xy_data(m, num)

            if (max_error is None) or (max_error_temp > max_error):
                max_error = max_error_temp
            else:
                pass

        # assign the determined index, lsq error and max. deviation from ideal function to the train object
        train_data.assign_ideal_func(ideal_index, total_error, max_error)

        # reset of variables after each train data column
        total_error = None
        max_error = 0
        ideal_index = None


def test_mapping(test_data, ideal_data, ideal_funcs_indices, ideal_funcs_errors):
    """
    mapping each data point of the test function to one of the 4 determined ideal functions
    If there are more than one possible mappings, the one with the smaller error will be selected
    :param test_data: DataNotIdeal object "test_data"
    :param ideal_data: Data object "ideal_data"
    :param ideal_funcs_indices: indices of the determined ideal functions
    :param ideal_funcs_errors: max errors between train data and determined ideal functions
    """
    # variable initialization
    mapped_function = []
    mapped_smallest_error = []
    i_temp = None
    error_lowest = 0

    # each row of the test data
    for i_row_test in range(0, test_data.return_num_rows()):
        # finding the correct x-value from the determined 4 ideal functions
        for i_row_ideal in range(0, ideal_data.return_num_rows()):
            if test_data.return_xy_data(i_row_test, 0) == ideal_data.return_xy_data(i_row_ideal, 0):
                # check for each determined ideal function if the mapping criterion is valid
                for i in range(0, len(ideal_funcs_indices)):
                    error_temp = abs(
                        ideal_data.return_xy_data(i_row_ideal, ideal_funcs_indices[i]) - test_data.return_xy_data(
                            i_row_test, 1))
                    # first ideal function
                    if i == 0:
                        error_lowest = error_temp
                        if error_lowest <= (ideal_funcs_errors[i] * np.sqrt(2)):
                            i_temp = i
                    # rest of the ideal functions
                    elif (i != 0) and (error_temp < error_lowest):
                        error_lowest = error_temp
                        if error_lowest <= (ideal_funcs_errors[i] * np.sqrt(2)):
                            i_temp = i

                mapped_function.append(i_temp)
                mapped_smallest_error.append(error_lowest)

                # reset of variables
                i_temp = None
                error_lowest = 0

    return mapped_function, mapped_smallest_error


def create_database(train_data, test_data, ideal_data, mapping, mapped_smallest_error, ideal_funcs_indices):
    """
    Function for creating the database where the data of the ideal functions, test functions
    and the found mapping from the test function to the determined ideal functions will be stored.
    With the same function the database will be loaded with the relevant data.
    :param train_data: DataNotIdeal object "train_data"
    :param test_data: DataNotIdeal object "test_data"
    :param ideal_data: Data object "ideal_data"
    :param mapping: determined mapping of each test data point to the determined ideal functions
    :param mapped_smallest_error: determined error between mapped test point and respective ideal function
    :param ideal_funcs_indices: indices of the determined ideal functions
    """
    # removes existing database, if there is already one existing
    if os.path.isfile('DATA.db'):
        os.remove('DATA.db')

    # general settings
    base_dir = os.getcwd()
    connection_string = "sqlite:///" + os.path.join(base_dir, 'DATA.db')
    engine = db.create_engine(connection_string, echo=True)
    connection = engine.connect()
    meta_data = db.MetaData()

    # creation of Table for training data sets for sql-database
    TrainingSets = db.Table(
        "Training_Sets", meta_data,
        db.Column("X", db.Float, nullable=False),
        db.Column("Y1", db.Float, nullable=False),
        db.Column("Y2", db.Float, nullable=False),
        db.Column("Y3", db.Float, nullable=False),
        db.Column("Y4", db.Float, nullable=False))

    # creation of Table for ideal functions data set for sql-database
    Ideal_functions = db.Table(
        "Ideal_functions", meta_data,
        db.Column("X", db.Float, nullable=False),
        db.Column("Y1", db.Float, nullable=False),
        db.Column("Y2", db.Float, nullable=False),
        db.Column("Y3", db.Float, nullable=False),
        db.Column("Y4", db.Float, nullable=False),
        db.Column("Y5", db.Float, nullable=False),
        db.Column("Y6", db.Float, nullable=False),
        db.Column("Y7", db.Float, nullable=False),
        db.Column("Y8", db.Float, nullable=False),
        db.Column("Y9", db.Float, nullable=False),
        db.Column("Y10", db.Float, nullable=False),
        db.Column("Y11", db.Float, nullable=False),
        db.Column("Y12", db.Float, nullable=False),
        db.Column("Y13", db.Float, nullable=False),
        db.Column("Y14", db.Float, nullable=False),
        db.Column("Y15", db.Float, nullable=False),
        db.Column("Y16", db.Float, nullable=False),
        db.Column("Y17", db.Float, nullable=False),
        db.Column("Y18", db.Float, nullable=False),
        db.Column("Y19", db.Float, nullable=False),
        db.Column("Y20", db.Float, nullable=False),
        db.Column("Y21", db.Float, nullable=False),
        db.Column("Y22", db.Float, nullable=False),
        db.Column("Y23", db.Float, nullable=False),
        db.Column("Y24", db.Float, nullable=False),
        db.Column("Y25", db.Float, nullable=False),
        db.Column("Y26", db.Float, nullable=False),
        db.Column("Y27", db.Float, nullable=False),
        db.Column("Y28", db.Float, nullable=False),
        db.Column("Y29", db.Float, nullable=False),
        db.Column("Y30", db.Float, nullable=False),
        db.Column("Y31", db.Float, nullable=False),
        db.Column("Y32", db.Float, nullable=False),
        db.Column("Y33", db.Float, nullable=False),
        db.Column("Y34", db.Float, nullable=False),
        db.Column("Y35", db.Float, nullable=False),
        db.Column("Y36", db.Float, nullable=False),
        db.Column("Y37", db.Float, nullable=False),
        db.Column("Y38", db.Float, nullable=False),
        db.Column("Y39", db.Float, nullable=False),
        db.Column("Y40", db.Float, nullable=False),
        db.Column("Y41", db.Float, nullable=False),
        db.Column("Y42", db.Float, nullable=False),
        db.Column("Y43", db.Float, nullable=False),
        db.Column("Y44", db.Float, nullable=False),
        db.Column("Y45", db.Float, nullable=False),
        db.Column("Y46", db.Float, nullable=False),
        db.Column("Y47", db.Float, nullable=False),
        db.Column("Y48", db.Float, nullable=False),
        db.Column("Y49", db.Float, nullable=False),
        db.Column("Y50", db.Float, nullable=False))

    # creation of Table for the found mapping
    Mapping = db.Table(
        "Mapping", meta_data,
        db.Column("X", db.Float, nullable=False),
        db.Column("Y", db.Float, nullable=False),
        db.Column("dY", db.String),
        db.Column("Ideal_function", db.String))

    # creation of all the tables in the sql-database
    meta_data.create_all(engine)

    # filling the sql table for the training data
    training_table = db.Table("Training_Sets", meta_data, autoload=True,
                              autoload_with=engine)
    sql_query_training = db.insert(training_table)

    data_list_training = []
    row_training = {}

    for i in range(0, train_data.return_num_rows()):
        for u in range(0, train_data.return_num_cols()):

            if u == 0:
                row_training = {'X': train_data.return_xy_data(i, u)}
            else:
                row_training['Y' + str(u)] = train_data.return_xy_data(i, u)

        data_list_training.append(row_training)
        # reset
        row_training = {}

    connection.execute(sql_query_training, data_list_training)

    # filling the sql table for the ideal functions
    ideal_func_table = db.Table("Ideal_functions", meta_data, autoload=True,
                                autoload_with=engine)
    sql_query_ideal_func = db.insert(ideal_func_table)

    data_list_ideal_func = []
    row_ideal = {}

    for i in range(0, ideal_data.return_num_rows()):
        for u in range(0, ideal_data.return_num_cols()):

            if u == 0:
                row_ideal = {'X': ideal_data.return_xy_data(i, u)}
            else:
                row_ideal['Y' + str(u)] = ideal_data.return_xy_data(i, u)

        data_list_ideal_func.append(row_ideal)
        # reset
        row_ideal = {}

    connection.execute(sql_query_ideal_func, data_list_ideal_func)

    # filling the sql table for the determined mapping
    mapping_table = db.Table("Mapping", meta_data, autoload=True,
                             autoload_with=engine)
    sql_query_mapping = db.insert(mapping_table)

    data_list_mapping = []
    row_mapping = {}

    for i in range(0, test_data.return_num_rows()):
        for u in range(0, 5):

            if u == 0:
                row_mapping = {'X': test_data.return_xy_data(i, u)}
            elif u == 1:
                row_mapping['Y'] = test_data.return_xy_data(i, u)
            elif u == 2:
                if mapping[i] is None:
                    row_mapping['dY'] = "None"
                else:
                    row_mapping['dY'] = str(mapped_smallest_error[i])
            elif u == 3:
                if mapping[i] is None:
                    row_mapping['Ideal_function'] = "None"
                else:
                    row_mapping['Ideal_function'] = str(ideal_funcs_indices[mapping[i]])

        data_list_mapping.append(row_mapping)
        # reset
        row_mapping = {}

    connection.execute(sql_query_mapping, data_list_mapping)


def extract_data(mapping, test_data, select_data):
    """
    Function for filtering out the relevant x and y data of the test function for visualization purposes
    :param mapping: determined mapping of each test data point to the determined ideal functions
    :param test_data: DataNotIdeal object "test_data"
    :param select_data: With the input parameter "select_data", the corresponding relevant data will be returned:
            1: x and y data of the test data which matches to the first determined ideal function
            2: x and y data of the test data which matches to the second determined ideal function
            3: x and y data of the test data which matches to the third determined ideal function
            4: x and y data of the test data which matches to the fourth determined ideal function
            "None": x and y data that could not be assigned to any ideal function
    """
    # creation of empty lists, in which the extracted data is stored
    matched_ideal4 = []
    matched_ideal3 = []
    matched_ideal2 = []
    matched_ideal1 = []
    matched_ideal_none = []
    x_values4 = []
    y_values4 = []
    x_values3 = []
    y_values3 = []
    x_values2 = []
    y_values2 = []
    x_values1 = []
    y_values1 = []
    x_values_none = []
    y_values_none = []

    # extract the relevant indices of the test data to the corresponding ideal function
    for i in range(0, len(mapping)):
        if mapping[i] == 3:
            matched_ideal4.append(i)
        elif mapping[i] == 2:
            matched_ideal3.append(i)
        elif mapping[i] == 1:
            matched_ideal2.append(i)
        elif mapping[i] == 0:
            matched_ideal1.append(i)
        else:
            matched_ideal_none.append(i)

    # adding the data to the corresponding list
    for i in matched_ideal4:
        x_values4.append(test_data.return_xy_data(i, 0))
        y_values4.append(test_data.return_xy_data(i, 1))
    for i in matched_ideal3:
        x_values3.append(test_data.return_xy_data(i, 0))
        y_values3.append(test_data.return_xy_data(i, 1))
    for i in matched_ideal2:
        x_values2.append(test_data.return_xy_data(i, 0))
        y_values2.append(test_data.return_xy_data(i, 1))
    for i in matched_ideal1:
        x_values1.append(test_data.return_xy_data(i, 0))
        y_values1.append(test_data.return_xy_data(i, 1))
    for i in matched_ideal_none:
        x_values_none.append(test_data.return_xy_data(i, 0))
        y_values_none.append(test_data.return_xy_data(i, 1))

    # returning the relevant data depending on the "select_data" input argument
    if select_data == 4:
        return x_values4, y_values4
    elif select_data == 3:
        return x_values3, y_values3
    elif select_data == 2:
        return x_values2, y_values2
    elif select_data == 1:
        return x_values1, y_values1
    elif select_data == "None":
        return x_values_none, y_values_none


def plot_ideal_functions(train_data, ideal_data, ideal_funcs_indices):
    """
    Function to plot all the results of the determination of the ideal data by minimizing the sum of errors with
    the help of the train data
    :param train_data: DataNotIdeal object "train_data"
    :param ideal_data: Data object "ideal_data"
    :param ideal_funcs_indices: indices of the determined ideal functions
    """
    # First test data
    plt.plot(train_data.return_x_data(), train_data.return_y_data(1), label="First train data")
    plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[0]), label="Ideal function")
    plt.legend()
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('Result of the ideal function determination')
    plt.show()

    # Second test data
    plt.plot(train_data.return_x_data(), train_data.return_y_data(2), label="Second train data")
    plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[1]), label="Ideal function")
    plt.legend()
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('Result of the ideal function determination')
    plt.show()

    # Third test data
    plt.plot(train_data.return_x_data(), train_data.return_y_data(3), label="Third train data")
    plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[2]), label="Ideal function")
    plt.legend()
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('Result of the ideal function determination')
    plt.show()

    # Fourth test data
    plt.plot(train_data.return_x_data(), train_data.return_y_data(4), label="Fourth train data")
    plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[3]), label="Ideal function")
    plt.legend()
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('Result of the ideal function determination')
    plt.show()


def plot_mapping(ideal_data, ideal_funcs_errors, ideal_funcs_indices, x_values, y_values, sel):
    """
    Function to plot all the results of the determination of the ideal data by minimizing the sum of errors with
    the help of the train data
    :param ideal_data: Data object "ideal_data"
    :param ideal_funcs_errors: max errors between train data and determined ideal functions
    :param ideal_funcs_indices: indices of the determined ideal functions
    :param x_values: Extracted x-values of the test data points assigned to the determined ideal function
    :param y_values: Extracted y-values of the test data points assigned to the determined ideal function
    :param sel: selection of which result of mapping to the ideal function should be plotted (1-4)
    """
    plt.plot(ideal_data.return_x_data(), ideal_data.return_y_data(ideal_funcs_indices[sel - 1]), label="Ideal function")
    plt.plot(ideal_data.return_x_data(),
             ideal_data.return_y_data(ideal_funcs_indices[sel - 1]) + (ideal_funcs_errors[sel - 1] * np.sqrt(2)), 'k--',
             label="error band")
    plt.plot(ideal_data.return_x_data(),
             ideal_data.return_y_data(ideal_funcs_indices[sel - 1]) - (ideal_funcs_errors[sel - 1] * np.sqrt(2)), 'k--',
             label="")
    plt.scatter(x_values, y_values, color='r', label="Mapped train data")
    plt.legend()
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')

    # changing the title of the plot depending on "sel" (the determined ideal function)
    if sel == 1:
        plt.title('Mapping of the first determined ideal function')
    elif sel == 2:
        plt.title('Mapping of the second determined ideal function')
    elif sel == 3:
        plt.title('Mapping of the third determined ideal function')
    elif sel == 4:
        plt.title('Mapping of the fourth determined ideal function')

    plt.show()


def plot_distribution(x1_len, x2_len, x3_len, x4_len, x_none_len):
    """
    Function to plot the distribution of the mapped train data points to the determined ideal functions
    :param x1_len: length of the x_value list of he test data mapped to the first determined ideal function
    :param x2_len: length of the x_value list of he test data mapped to the second determined ideal function
    :param x3_len: length of the x_value list of he test data mapped to the third determined ideal function
    :param x4_len: length of the x_value list of he test data mapped to the fourth determined ideal function
    :param x_none_len: length of the x_value list of he test data not mapped to any ideal function
    """
    data_percentage = [x1_len, x2_len, x3_len, x4_len, x_none_len]
    my_labels = ["first", "second", "third", "fourth", "not mapped"]
    plt.pie(data_percentage, labels=my_labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title("Mapping distribution of the test data points")
    plt.show()
