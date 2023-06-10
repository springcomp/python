#!/usr/bin/env python

import argparse
import sys
from typing import List
import numpy as np
import os
import pandas as pd
import random
import time

from experiment.utils import pad_max_length_arrays
from experiment.utils import has_dict_key
from experiment.utils import stderr_exit

from sklearn.linear_model import TheilSenRegressor
from sklearn.model_selection import train_test_split


def linear_regression(X, Y):
    # Add a column of ones to X for the intercept term
    X = np.column_stack((np.ones(len(X)), X))

    # Calculate the coefficients using least squares method
    coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
    intercept_b = coefficients[0]
    coefficient_a = coefficients[1:]

    # Calculate the predicted values
    Y_pred = np.dot(X, coefficients)

    # Calculate the explained sum of squares (ESS)
    ESS = np.sum((Y_pred - np.mean(Y)) ** 2)

    # Calculate the total sum of squares (TSS)
    TSS = np.sum((Y - np.mean(Y)) ** 2)

    # Calculate the coefficient of determination (R^2)
    if TSS == 0:
        r_squared = 0
    else:
        r_squared = ESS / TSS

    return intercept_b, coefficient_a, r_squared


def parabolic_regression(X, Y):
    # Add a column of ones to X for the intercept term
    X = np.column_stack((np.ones(len(X)), X, X**2))

    # Calculate the coefficients using least squares method
    coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
    intercept_c = coefficients[0]
    coefficient_b = coefficients[1:]
    coefficient_a = coefficients[2:]

    # Calculate the predicted values
    Y_pred = np.dot(X, coefficients)

    # Calculate the mean squared error (MSE)
    mse = np.mean((Y - Y_pred) ** 2)

    return intercept_c, coefficient_b, coefficient_a, mse


def cubic_regression(X, Y):
    # Add a column of ones to X for the intercept term
    X = np.column_stack((np.ones(len(X)), X, X**2, X**3))

    # Calculate the coefficients using least squares method
    coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
    intercept_d = coefficients[0]
    coefficient_c = coefficients[1:]
    coefficient_b = coefficients[2:]
    coefficient_a = coefficients[3:]

    # Calculate the predicted values
    Y_pred = np.dot(X, coefficients)

    # Calculate the mean squared error (MSE)
    mse = np.mean((Y - Y_pred) ** 2)

    return intercept_d, coefficient_c, coefficient_b, coefficient_a, mse


def theil_sen_regression(X, Y):
    # Create an instance of TheilSenRegressor
    regressor = TheilSenRegressor()

    # Fit the regressor to the data
    regressor.fit(X, Y)

    # Retrieve the coefficients
    coefficient_a = regressor.coef_
    intercept_b = regressor.intercept_

    # Calculate the coefficient of determination (R^2)
    r_squared = regressor.score(X, Y)

    return intercept_b, coefficient_a, r_squared


def calculate_mse(model, actual):
    if model.shape != actual.shape:
        raise ValueError("Model and actual arrays must have the same shape.")
    return np.mean((model - actual) ** 2)

# Function to measure the time


def measure_time(func):
    start_time = time.time()  # Record the start time

    # Call the provided function
    func()

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    return elapsed_time


def run(args: dict) -> None:
    # Create a dictionary with your analysis results
    results = {
        'Datasets': [],
        'Linear_a_speed_1': [],
        'Linear_a_oilp_1': [],
        'Linear_a_cool_1': [],
        'Linear_a_boost_1': [],
        'Linear_a_temp1_1': [],
        'Linear_a_temp2_1': [],
        'Linear_a_oilt_1': [],
        'Linear_a_fahrt_1': [],
        'Linear_a_wind_1': [],
        'Linear_a_clock_1': [],
        'Linear_b_1': [],
        'Linear_rtwo_1': [],
        'Linear_trainingtime_1': [],
        'Linear_test_MSE_1': [],
        'Linear_testtime_1': [],
        'Sen_a_speed_1': [],
        'Sen_a_oilp_1': [],
        'Sen_a_cool_1': [],
        'Sen_a_boost_1': [],
        'Sen_a_temp1_1': [],
        'Sen_a_temp2_1': [],
        'Sen_a_oilt_1': [],
        'Sen_a_fahrt_1': [],
        'Sen_a_wind_1': [],
        'Sen_a_clock_1': [],
        'Sen_b_1': [],
        'Sen_rtwo_1': [],
        'Sen_trainingtime_1': [],
        'Sen_test_MSE_1': [],
        'Sen_testtime_1': [],
        'Parabolic_a_speed_1': [],
        'Parabolic_a_oilp_1': [],
        'Parabolic_a_cool_1': [],
        'Parabolic_a_boost_1': [],
        'Parabolic_a_temp1_1': [],
        'Parabolic_a_temp2_1': [],
        'Parabolic_a_oilt_1': [],
        'Parabolic_a_fahrt_1': [],
        'Parabolic_a_wind_1': [],
        'Parabolic_a_clock_1': [],
        'Parabolic_b_speed_1': [],
        'Parabolic_b_oilp_1': [],
        'Parabolic_b_cool_1': [],
        'Parabolic_b_boost_1': [],
        'Parabolic_b_temp1_1': [],
        'Parabolic_b_temp2_1': [],
        'Parabolic_b_oilt_1': [],
        'Parabolic_b_fahrt_1': [],
        'Parabolic_b_wind_1': [],
        'Parabolic_b_clock_1': [],
        'Parabolic_c_1': [],
        'Parabolic_MSE_1': [],
        'Parabolic_trainingtime_1': [],
        'Parabolic_test_MSE_1': [],
        'Parabolic_testtime_1': [],
        'Cubic_a_speed_1': [],
        'Cubic_a_oilp_1': [],
        'Cubic_a_cool_1': [],
        'Cubic_a_boost_1': [],
        'Cubic_a_temp1_1': [],
        'Cubic_a_temp2_1': [],
        'Cubic_a_oilt_1': [],
        'Cubic_a_fahrt_1': [],
        'Cubic_a_wind_1': [],
        'Cubic_a_clock_1': [],
        'Cubic_b_speed_1': [],
        'Cubic_b_oilp_1': [],
        'Cubic_b_cool_1': [],
        'Cubic_b_boost_1': [],
        'Cubic_b_temp1_1': [],
        'Cubic_b_temp2_1': [],
        'Cubic_b_oilt_1': [],
        'Cubic_b_fahrt_1': [],
        'Cubic_b_wind_1': [],
        'Cubic_b_clock_1': [],
        'Cubic_c_speed_1': [],
        'Cubic_c_oilp_1': [],
        'Cubic_c_cool_1': [],
        'Cubic_c_boost_1': [],
        'Cubic_c_temp1_1': [],
        'Cubic_c_temp2_1': [],
        'Cubic_c_oilt_1': [],
        'Cubic_c_fahrt_1': [],
        'Cubic_c_wind_1': [],
        'Cubic_c_clock_1': [],
        'Cubic_d_1': [],
        'Cubic_MSE_1': [],
        'Cubic_trainingtime_1': [],
        'Cubic_test_MSE_1': [],
        'Cubic_testtime_1': [],
        'Linear_a_speed_2': [],
        'Linear_a_oilp_2': [],
        'Linear_a_cool_2': [],
        'Linear_a_boost_2': [],
        'Linear_a_temp1_2': [],
        'Linear_a_temp2_2': [],
        'Linear_a_oilt_2': [],
        'Linear_a_fahrt_2': [],
        'Linear_a_wind_2': [],
        'Linear_a_clock_2': [],
        'Linear_b_2': [],
        'Linear_rtwo_2': [],
        'Linear_trainingtime_2': [],
        'Linear_test_MSE_2': [],
        'Linear_testtime_2': [],
        'Sen_a_speed_2': [],
        'Sen_a_oilp_2': [],
        'Sen_a_cool_2': [],
        'Sen_a_boost_2': [],
        'Sen_a_temp1_2': [],
        'Sen_a_temp2_2': [],
        'Sen_a_oilt_2': [],
        'Sen_a_fahrt_2': [],
        'Sen_a_wind_2': [],
        'Sen_a_clock_2': [],
        'Sen_b_2': [],
        'Sen_rtwo_2': [],
        'Sen_trainingtime_2': [],
        'Sen_test_MSE_2': [],
        'Sen_testtime_2': [],
        'Parabolic_a_speed_2': [],
        'Parabolic_a_oilp_2': [],
        'Parabolic_a_cool_2': [],
        'Parabolic_a_boost_2': [],
        'Parabolic_a_temp1_2': [],
        'Parabolic_a_temp2_2': [],
        'Parabolic_a_oilt_2': [],
        'Parabolic_a_fahrt_2': [],
        'Parabolic_a_wind_2': [],
        'Parabolic_a_clock_2': [],
        'Parabolic_b_speed_2': [],
        'Parabolic_b_oilp_2': [],
        'Parabolic_b_cool_2': [],
        'Parabolic_b_boost_2': [],
        'Parabolic_b_temp1_2': [],
        'Parabolic_b_temp2_2': [],
        'Parabolic_b_oilt_2': [],
        'Parabolic_b_fahrt_2': [],
        'Parabolic_b_wind_2': [],
        'Parabolic_b_clock_2': [],
        'Parabolic_c_2': [],
        'Parabolic_MSE_2': [],
        'Parabolic_trainingtime_2': [],
        'Parabolic_test_MSE_2': [],
        'Parabolic_testtime_2': [],
        'Cubic_a_speed_2': [],
        'Cubic_a_oilp_2': [],
        'Cubic_a_cool_2': [],
        'Cubic_a_boost_2': [],
        'Cubic_a_temp1_2': [],
        'Cubic_a_temp2_2': [],
        'Cubic_a_oilt_2': [],
        'Cubic_a_fahrt_2': [],
        'Cubic_a_wind_2': [],
        'Cubic_a_clock_2': [],
        'Cubic_b_speed_2': [],
        'Cubic_b_oilp_2': [],
        'Cubic_b_cool_2': [],
        'Cubic_b_boost_2': [],
        'Cubic_b_temp1_2': [],
        'Cubic_b_temp2_2': [],
        'Cubic_b_oilt_2': [],
        'Cubic_b_fahrt_2': [],
        'Cubic_b_wind_2': [],
        'Cubic_b_clock_2': [],
        'Cubic_c_speed_2': [],
        'Cubic_c_oilp_2': [],
        'Cubic_c_cool_2': [],
        'Cubic_c_boost_2': [],
        'Cubic_c_temp1_2': [],
        'Cubic_c_temp2_2': [],
        'Cubic_c_oilt_2': [],
        'Cubic_c_fahrt_2': [],
        'Cubic_c_wind_2': [],
        'Cubic_c_clock_2': [],
        'Cubic_d_2': [],
        'Cubic_MSE_2': [],
        'Cubic_trainingtime_2': [],
        'Cubic_test_MSE_2': [],
        'Cubic_testtime_2': [],
        'Linear_a_speed_3': [],
        'Linear_a_oilp_3': [],
        'Linear_a_cool_3': [],
        'Linear_a_boost_3': [],
        'Linear_a_temp1_3': [],
        'Linear_a_temp2_3': [],
        'Linear_a_oilt_3': [],
        'Linear_a_fahrt_3': [],
        'Linear_a_wind_3': [],
        'Linear_a_clock_3': [],
        'Linear_b_3': [],
        'Linear_rtwo_3': [],
        'Linear_trainingtime_3': [],
        'Linear_test_MSE_3': [],
        'Linear_testtime_3': [],
        'Sen_a_speed_3': [],
        'Sen_a_oilp_3': [],
        'Sen_a_cool_3': [],
        'Sen_a_boost_3': [],
        'Sen_a_temp1_3': [],
        'Sen_a_temp2_3': [],
        'Sen_a_oilt_3': [],
        'Sen_a_fahrt_3': [],
        'Sen_a_wind_3': [],
        'Sen_a_clock_3': [],
        'Sen_b_3': [],
        'Sen_rtwo_3': [],
        'Sen_trainingtime_3': [],
        'Sen_test_MSE_3': [],
        'Sen_testtime_3': [],
        'Parabolic_a_speed_3': [],
        'Parabolic_a_oilp_3': [],
        'Parabolic_a_cool_3': [],
        'Parabolic_a_boost_3': [],
        'Parabolic_a_temp1_3': [],
        'Parabolic_a_temp2_3': [],
        'Parabolic_a_oilt_3': [],
        'Parabolic_a_fahrt_3': [],
        'Parabolic_a_wind_3': [],
        'Parabolic_a_clock_3': [],
        'Parabolic_b_speed_3': [],
        'Parabolic_b_oilp_3': [],
        'Parabolic_b_cool_3': [],
        'Parabolic_b_boost_3': [],
        'Parabolic_b_temp1_3': [],
        'Parabolic_b_temp2_3': [],
        'Parabolic_b_oilt_3': [],
        'Parabolic_b_fahrt_3': [],
        'Parabolic_b_wind_3': [],
        'Parabolic_b_clock_3': [],
        'Parabolic_c_3': [],
        'Parabolic_MSE_3': [],
        'Parabolic_trainingtime_3': [],
        'Parabolic_test_MSE_3': [],
        'Parabolic_testtime_3': [],
        'Cubic_a_speed_3': [],
        'Cubic_a_oilp_3': [],
        'Cubic_a_cool_3': [],
        'Cubic_a_boost_3': [],
        'Cubic_a_temp1_3': [],
        'Cubic_a_temp2_3': [],
        'Cubic_a_oilt_3': [],
        'Cubic_a_fahrt_3': [],
        'Cubic_a_wind_3': [],
        'Cubic_a_clock_3': [],
        'Cubic_b_speed_3': [],
        'Cubic_b_oilp_3': [],
        'Cubic_b_cool_3': [],
        'Cubic_b_boost_3': [],
        'Cubic_b_temp1_3': [],
        'Cubic_b_temp2_3': [],
        'Cubic_b_oilt_3': [],
        'Cubic_b_fahrt_3': [],
        'Cubic_b_wind_3': [],
        'Cubic_b_clock_3': [],
        'Cubic_c_speed_3': [],
        'Cubic_c_oilp_3': [],
        'Cubic_c_cool_3': [],
        'Cubic_c_boost_3': [],
        'Cubic_c_temp1_3': [],
        'Cubic_c_temp2_3': [],
        'Cubic_c_oilt_3': [],
        'Cubic_c_fahrt_3': [],
        'Cubic_c_wind_3': [],
        'Cubic_c_clock_3': [],
        'Cubic_d_3': [],
        'Cubic_MSE_3': [],
        'Cubic_trainingtime_3': [],
        'Cubic_test_MSE_3': [],
        'Cubic_testtime_3': [],
    }

    # Loop through each dataset CSV file

    dataset_files = args['dataset_files']
    for i in range(len(dataset_files)):
        csv_file = dataset_files[i]

        # Read the CSV file
        data = pd.read_csv(csv_file, sep=";")

        # Split the data into training and testing datasets
        train_data, test_data = train_test_split(
            data, test_size=0.4, random_state=74)

        Time = train_data['Time']  # Time
        SpeedKmHr = train_data['SpeedKmHr']  # 63
        Wind = train_data['Wind']
        Wind_Speed = train_data['Wind_Speed']  # 16
        Engine1_engine_speed = train_data['Engine1_engine_speed']  # 32
        # 69
        Engine1_lube_oil_pressure = train_data['Engine1_lube_oil_pressure']
        # 28
        Engine1_coolant_temperature = train_data['Engine1_coolant_temperature']
        Engine1_boost_pressure = train_data['Engine1_boost_pressure']  # 41
        Engine1_engine_load = train_data['Engine1_engine_load']  # 5
        # 12
        Engine1_exhaust_temperature1 = train_data['Engine1_exhaust_temperature1']
        # 29
        Engine1_exhaust_temperature2 = train_data['Engine1_exhaust_temperature2']
        Engine1_fuel_consumption = train_data['Engine1_fuel_consumption']  # 7
        # 13
        Engine1_lube_oil_temperature = train_data['Engine1_lube_oil_temperature']
        Engine2_engine_speed = train_data['Engine2_engine_speed']  # 8
        # 42
        Engine2_lube_oil_pressure = train_data['Engine2_lube_oil_pressure']
        # 51
        Engine2_coolant_temperature = train_data['Engine2_coolant_temperature']
        Engine2_boost_pressure = train_data['Engine2_boost_pressure']  # 1
        Engine2_engine_load = train_data['Engine2_engine_load']  # 9
        # 26
        Engine2_exhaust_temperature1 = train_data['Engine2_exhaust_temperature1']
        # 61
        Engine2_exhaust_temperature2 = train_data['Engine2_exhaust_temperature2']
        Engine2_fuel_consumption = train_data['Engine2_fuel_consumption']  # 52
        # 22
        Engine2_lube_oil_temperature = train_data['Engine2_lube_oil_temperature']
        Engine3_engine_speed = train_data['Engine3_engine_speed']  # 14
        # 56
        Engine3_lube_oil_pressure = train_data['Engine3_lube_oil_pressure']
        # 57
        Engine3_coolant_temperature = train_data['Engine3_coolant_temperature']
        Engine3_boost_pressure = train_data['Engine3_boost_pressure']  # 33
        Engine3_engine_load = train_data['Engine3_engine_load']  # 30
        # 20
        Engine3_exhaust_temperature1 = train_data['Engine3_exhaust_temperature1']
        # 15
        Engine3_exhaust_temperature2 = train_data['Engine3_exhaust_temperature2']
        Engine3_fuel_consumption = train_data['Engine3_fuel_consumption']  # 62
        # 53
        Engine3_lube_oil_temperature = train_data['Engine3_lube_oil_temperature']
        Clock = train_data['Clock']

        # Prepare the data
        x1 = np.vstack((Engine1_engine_speed, Engine1_lube_oil_pressure, Engine1_coolant_temperature, Engine1_boost_pressure,
                        Engine1_exhaust_temperature1, Engine1_exhaust_temperature2, Engine1_lube_oil_temperature, SpeedKmHr, Wind, Clock)).T
        x2 = np.vstack((Engine2_engine_speed, Engine2_lube_oil_pressure, Engine2_coolant_temperature, Engine2_boost_pressure,
                        Engine2_exhaust_temperature1, Engine2_exhaust_temperature2, Engine2_lube_oil_temperature, SpeedKmHr, Wind, Clock)).T
        x3 = np.vstack((Engine3_engine_speed, Engine3_lube_oil_pressure, Engine3_coolant_temperature, Engine3_boost_pressure,
                        Engine3_exhaust_temperature1, Engine3_exhaust_temperature2, Engine3_lube_oil_temperature, SpeedKmHr, Wind, Clock)).T
        y1 = Engine1_engine_load
        y2 = Engine2_engine_load
        y3 = Engine3_engine_load

        # Extract the filename from the path
        filename = os.path.basename(csv_file)

        # Append the extracted content to the 'Datasets' array in the 'results' dictionary
        results['Datasets'].append(filename)

        # Analysis for Engine 1

        # Calculate linear regression
        linear_b, linear_a, linear_rtwo = linear_regression(x1, y1)

        # Assign coefficients to their respective columns
        results['Linear_a_speed_1'].extend([linear_a[0]])
        results['Linear_a_oilp_1'].extend([linear_a[1]])
        results['Linear_a_cool_1'].extend([linear_a[2]])
        results['Linear_a_boost_1'].extend([linear_a[3]])
        results['Linear_a_temp1_1'].extend([linear_a[4]])
        results['Linear_a_temp2_1'].extend([linear_a[5]])
        results['Linear_a_oilt_1'].extend([linear_a[6]])
        results['Linear_a_fahrt_1'].extend([linear_a[7]])
        results['Linear_a_wind_1'].extend([linear_a[8]])
        results['Linear_a_clock_1'].extend([linear_a[9]])
        results['Linear_b_1'].extend([linear_b])
        results['Linear_rtwo_1'].extend([linear_rtwo])

        # Calculate Theil-Sen regression
        sen_b, sen_a, sen_rtwo = theil_sen_regression(x1, y1)

        # Assign coefficients to their respective columns
        results['Sen_a_speed_1'].extend([sen_a[0]])
        results['Sen_a_oilp_1'].extend([sen_a[1]])
        results['Sen_a_cool_1'].extend([sen_a[2]])
        results['Sen_a_boost_1'].extend([sen_a[3]])
        results['Sen_a_temp1_1'].extend([sen_a[4]])
        results['Sen_a_temp2_1'].extend([sen_a[5]])
        results['Sen_a_oilt_1'].extend([sen_a[6]])
        results['Sen_a_fahrt_1'].extend([sen_a[7]])
        results['Sen_a_wind_1'].extend([sen_a[8]])
        results['Sen_a_clock_1'].extend([sen_a[9]])
        results['Sen_b_1'].extend([sen_b])
        results['Sen_rtwo_1'].extend([sen_rtwo])

        # Calculate parabolic regression
        parabolic_c, parabolic_b, parabolic_a, parabolic_mse = parabolic_regression(
            x1, y1)

        # Appending values to the arrays
        results['Parabolic_a_speed_1'].extend([parabolic_a[0]])
        results['Parabolic_a_oilp_1'].extend([parabolic_a[1]])
        results['Parabolic_a_cool_1'].extend([parabolic_a[2]])
        results['Parabolic_a_boost_1'].extend([parabolic_a[3]])
        results['Parabolic_a_temp1_1'].extend([parabolic_a[4]])
        results['Parabolic_a_temp2_1'].extend([parabolic_a[5]])
        results['Parabolic_a_oilt_1'].extend([parabolic_a[6]])
        results['Parabolic_a_fahrt_1'].extend([parabolic_a[7]])
        results['Parabolic_a_wind_1'].extend([parabolic_a[8]])
        results['Parabolic_a_clock_1'].extend([parabolic_a[9]])
        results['Parabolic_b_speed_1'].extend([parabolic_b[0]])
        results['Parabolic_b_oilp_1'].extend([parabolic_b[1]])
        results['Parabolic_b_cool_1'].extend([parabolic_b[2]])
        results['Parabolic_b_boost_1'].extend([parabolic_b[3]])
        results['Parabolic_b_temp1_1'].extend([parabolic_b[4]])
        results['Parabolic_b_temp2_1'].extend([parabolic_b[5]])
        results['Parabolic_b_oilt_1'].extend([parabolic_b[6]])
        results['Parabolic_b_fahrt_1'].extend([parabolic_b[7]])
        results['Parabolic_b_wind_1'].extend([parabolic_b[8]])
        results['Parabolic_b_clock_1'].extend([parabolic_b[9]])
        results['Parabolic_c_1'].extend([parabolic_c])
        results['Parabolic_MSE_1'].extend([parabolic_mse])

        # Calculate cubic regression
        cubic_d, cubic_c, cubic_b, cubic_a, cubic_mse = cubic_regression(
            x1, y1)

        # Appending values to the arrays
        results['Cubic_a_speed_1'].extend([cubic_a[0]])
        results['Cubic_a_oilp_1'].extend([cubic_a[1]])
        results['Cubic_a_cool_1'].extend([cubic_a[2]])
        results['Cubic_a_boost_1'].extend([cubic_a[3]])
        results['Cubic_a_temp1_1'].extend([cubic_a[4]])
        results['Cubic_a_temp2_1'].extend([cubic_a[5]])
        results['Cubic_a_oilt_1'].extend([cubic_a[6]])
        results['Cubic_a_fahrt_1'].extend([cubic_a[7]])
        results['Cubic_a_wind_1'].extend([cubic_a[8]])
        results['Cubic_a_clock_1'].extend([cubic_a[9]])
        results['Cubic_b_speed_1'].extend([cubic_b[0]])
        results['Cubic_b_oilp_1'].extend([cubic_b[1]])
        results['Cubic_b_cool_1'].extend([cubic_b[2]])
        results['Cubic_b_boost_1'].extend([cubic_b[3]])
        results['Cubic_b_temp1_1'].extend([cubic_b[4]])
        results['Cubic_b_temp2_1'].extend([cubic_b[5]])
        results['Cubic_b_oilt_1'].extend([cubic_b[6]])
        results['Cubic_b_fahrt_1'].extend([cubic_b[7]])
        results['Cubic_b_wind_1'].extend([cubic_b[8]])
        results['Cubic_b_clock_1'].extend([cubic_b[9]])
        results['Cubic_c_speed_1'].extend([cubic_c[0]])
        results['Cubic_c_oilp_1'].extend([cubic_c[1]])
        results['Cubic_c_cool_1'].extend([cubic_c[2]])
        results['Cubic_c_boost_1'].extend([cubic_c[3]])
        results['Cubic_c_temp1_1'].extend([cubic_c[4]])
        results['Cubic_c_temp2_1'].extend([cubic_c[5]])
        results['Cubic_c_oilt_1'].extend([cubic_c[6]])
        results['Cubic_c_fahrt_1'].extend([cubic_c[7]])
        results['Cubic_c_wind_1'].extend([cubic_c[8]])
        results['Cubic_c_clock_1'].extend([cubic_c[9]])
        results['Cubic_d_1'].extend([cubic_d])
        results['Cubic_MSE_1'].extend([cubic_mse])

        # Analysis for Engine 2

        # Calculate linear regression
        linear_b, linear_a, linear_rtwo = linear_regression(x2, y2)

        # Assign coefficients to their respective columns
        results['Linear_a_speed_2'].extend([linear_a[0]])
        results['Linear_a_oilp_2'].extend([linear_a[1]])
        results['Linear_a_cool_2'].extend([linear_a[2]])
        results['Linear_a_boost_2'].extend([linear_a[3]])
        results['Linear_a_temp1_2'].extend([linear_a[4]])
        results['Linear_a_temp2_2'].extend([linear_a[5]])
        results['Linear_a_oilt_2'].extend([linear_a[6]])
        results['Linear_a_fahrt_2'].extend([linear_a[7]])
        results['Linear_a_wind_2'].extend([linear_a[8]])
        results['Linear_a_clock_2'].extend([linear_a[9]])
        results['Linear_b_2'].extend([linear_b])
        results['Linear_rtwo_2'].extend([linear_rtwo])

        # Calculate Theil-Sen regression
        sen_b, sen_a, sen_rtwo = theil_sen_regression(x2, y2)

        # Assign coefficients to their respective columns
        results['Sen_a_speed_2'].extend([sen_a[0]])
        results['Sen_a_oilp_2'].extend([sen_a[1]])
        results['Sen_a_cool_2'].extend([sen_a[2]])
        results['Sen_a_boost_2'].extend([sen_a[3]])
        results['Sen_a_temp1_2'].extend([sen_a[4]])
        results['Sen_a_temp2_2'].extend([sen_a[5]])
        results['Sen_a_oilt_2'].extend([sen_a[6]])
        results['Sen_a_fahrt_2'].extend([sen_a[7]])
        results['Sen_a_wind_2'].extend([sen_a[8]])
        results['Sen_a_clock_2'].extend([sen_a[9]])
        results['Sen_b_2'].extend([sen_b])
        results['Sen_rtwo_2'].extend([sen_rtwo])

        # Calculate parabolic regression
        parabolic_c, parabolic_b, parabolic_a, parabolic_mse = parabolic_regression(
            x2, y2)

        # Appending values to the arrays
        results['Parabolic_a_speed_2'].extend([parabolic_a[0]])
        results['Parabolic_a_oilp_2'].extend([parabolic_a[1]])
        results['Parabolic_a_cool_2'].extend([parabolic_a[2]])
        results['Parabolic_a_boost_2'].extend([parabolic_a[3]])
        results['Parabolic_a_temp1_2'].extend([parabolic_a[4]])
        results['Parabolic_a_temp2_2'].extend([parabolic_a[5]])
        results['Parabolic_a_oilt_2'].extend([parabolic_a[6]])
        results['Parabolic_a_fahrt_2'].extend([parabolic_a[7]])
        results['Parabolic_a_wind_2'].extend([parabolic_a[8]])
        results['Parabolic_a_clock_2'].extend([parabolic_a[9]])
        results['Parabolic_b_speed_2'].extend([parabolic_b[0]])
        results['Parabolic_b_oilp_2'].extend([parabolic_b[1]])
        results['Parabolic_b_cool_2'].extend([parabolic_b[2]])
        results['Parabolic_b_boost_2'].extend([parabolic_b[3]])
        results['Parabolic_b_temp1_2'].extend([parabolic_b[4]])
        results['Parabolic_b_temp2_2'].extend([parabolic_b[5]])
        results['Parabolic_b_oilt_2'].extend([parabolic_b[6]])
        results['Parabolic_b_fahrt_2'].extend([parabolic_b[7]])
        results['Parabolic_b_wind_2'].extend([parabolic_b[8]])
        results['Parabolic_b_clock_2'].extend([parabolic_b[9]])
        results['Parabolic_c_2'].extend([parabolic_c])
        results['Parabolic_MSE_2'].extend([parabolic_mse])

        # Calculate cubic regression
        cubic_d, cubic_c, cubic_b, cubic_a, cubic_mse = cubic_regression(
            x2, y2)

        # Appending values to the arrays
        results['Cubic_a_speed_2'].extend([cubic_a[0]])
        results['Cubic_a_oilp_2'].extend([cubic_a[1]])
        results['Cubic_a_cool_2'].extend([cubic_a[2]])
        results['Cubic_a_boost_2'].extend([cubic_a[3]])
        results['Cubic_a_temp1_2'].extend([cubic_a[4]])
        results['Cubic_a_temp2_2'].extend([cubic_a[5]])
        results['Cubic_a_oilt_2'].extend([cubic_a[6]])
        results['Cubic_a_fahrt_2'].extend([cubic_a[7]])
        results['Cubic_a_wind_2'].extend([cubic_a[8]])
        results['Cubic_a_clock_2'].extend([cubic_a[9]])
        results['Cubic_b_speed_2'].extend([cubic_b[0]])
        results['Cubic_b_oilp_2'].extend([cubic_b[1]])
        results['Cubic_b_cool_2'].extend([cubic_b[2]])
        results['Cubic_b_boost_2'].extend([cubic_b[3]])
        results['Cubic_b_temp1_2'].extend([cubic_b[4]])
        results['Cubic_b_temp2_2'].extend([cubic_b[5]])
        results['Cubic_b_oilt_2'].extend([cubic_b[6]])
        results['Cubic_b_fahrt_2'].extend([cubic_b[7]])
        results['Cubic_b_wind_2'].extend([cubic_b[8]])
        results['Cubic_b_clock_2'].extend([cubic_b[9]])
        results['Cubic_c_speed_2'].extend([cubic_c[0]])
        results['Cubic_c_oilp_2'].extend([cubic_c[1]])
        results['Cubic_c_cool_2'].extend([cubic_c[2]])
        results['Cubic_c_boost_2'].extend([cubic_c[3]])
        results['Cubic_c_temp1_2'].extend([cubic_c[4]])
        results['Cubic_c_temp2_2'].extend([cubic_c[5]])
        results['Cubic_c_oilt_2'].extend([cubic_c[6]])
        results['Cubic_c_fahrt_2'].extend([cubic_c[7]])
        results['Cubic_c_wind_2'].extend([cubic_c[8]])
        results['Cubic_c_clock_2'].extend([cubic_c[9]])
        results['Cubic_d_2'].extend([cubic_d])
        results['Cubic_MSE_2'].extend([cubic_mse])

        # Analysis for Engine 3

        # Calculate linear regression
        linear_b, linear_a, linear_rtwo = linear_regression(x3, y3)

        # Assign coefficients to their respective columns
        results['Linear_a_speed_3'].extend([linear_a[0]])
        results['Linear_a_oilp_3'].extend([linear_a[1]])
        results['Linear_a_cool_3'].extend([linear_a[2]])
        results['Linear_a_boost_3'].extend([linear_a[3]])
        results['Linear_a_temp1_3'].extend([linear_a[4]])
        results['Linear_a_temp2_3'].extend([linear_a[5]])
        results['Linear_a_oilt_3'].extend([linear_a[6]])
        results['Linear_a_fahrt_3'].extend([linear_a[7]])
        results['Linear_a_wind_3'].extend([linear_a[8]])
        results['Linear_a_clock_3'].extend([linear_a[9]])
        results['Linear_b_3'].extend([linear_b])
        results['Linear_rtwo_3'].extend([linear_rtwo])

        # Calculate Theil-Sen regression
        sen_b, sen_a, sen_rtwo = theil_sen_regression(x3, y3)

        # Assign coefficients to their respective columns
        results['Sen_a_speed_3'].extend([sen_a[0]])
        results['Sen_a_oilp_3'].extend([sen_a[1]])
        results['Sen_a_cool_3'].extend([sen_a[2]])
        results['Sen_a_boost_3'].extend([sen_a[3]])
        results['Sen_a_temp1_3'].extend([sen_a[4]])
        results['Sen_a_temp2_3'].extend([sen_a[5]])
        results['Sen_a_oilt_3'].extend([sen_a[6]])
        results['Sen_a_fahrt_3'].extend([sen_a[7]])
        results['Sen_a_wind_3'].extend([sen_a[8]])
        results['Sen_a_clock_3'].extend([sen_a[9]])
        results['Sen_b_3'].extend([sen_b])
        results['Sen_rtwo_3'].extend([sen_rtwo])

        # Calculate parabolic regression
        parabolic_c, parabolic_b, parabolic_a, parabolic_mse = parabolic_regression(
            x3, y3)

        # Appending values to the arrays
        results['Parabolic_a_speed_3'].extend([parabolic_a[0]])
        results['Parabolic_a_oilp_3'].extend([parabolic_a[1]])
        results['Parabolic_a_cool_3'].extend([parabolic_a[2]])
        results['Parabolic_a_boost_3'].extend([parabolic_a[3]])
        results['Parabolic_a_temp1_3'].extend([parabolic_a[4]])
        results['Parabolic_a_temp2_3'].extend([parabolic_a[5]])
        results['Parabolic_a_oilt_3'].extend([parabolic_a[6]])
        results['Parabolic_a_fahrt_3'].extend([parabolic_a[7]])
        results['Parabolic_a_wind_3'].extend([parabolic_a[8]])
        results['Parabolic_a_clock_3'].extend([parabolic_a[9]])
        results['Parabolic_b_speed_3'].extend([parabolic_b[0]])
        results['Parabolic_b_oilp_3'].extend([parabolic_b[1]])
        results['Parabolic_b_cool_3'].extend([parabolic_b[2]])
        results['Parabolic_b_boost_3'].extend([parabolic_b[3]])
        results['Parabolic_b_temp1_3'].extend([parabolic_b[4]])
        results['Parabolic_b_temp2_3'].extend([parabolic_b[5]])
        results['Parabolic_b_oilt_3'].extend([parabolic_b[6]])
        results['Parabolic_b_fahrt_3'].extend([parabolic_b[7]])
        results['Parabolic_b_wind_3'].extend([parabolic_b[8]])
        results['Parabolic_b_clock_3'].extend([parabolic_b[9]])
        results['Parabolic_c_3'].extend([parabolic_c])
        results['Parabolic_MSE_3'].extend([parabolic_mse])

        # Calculate cubic regression
        cubic_d, cubic_c, cubic_b, cubic_a, cubic_mse = cubic_regression(
            x3, y3)

        # Appending values to the arrays
        results['Cubic_a_speed_3'].extend([cubic_a[0]])
        results['Cubic_a_oilp_3'].extend([cubic_a[1]])
        results['Cubic_a_cool_3'].extend([cubic_a[2]])
        results['Cubic_a_boost_3'].extend([cubic_a[3]])
        results['Cubic_a_temp1_3'].extend([cubic_a[4]])
        results['Cubic_a_temp2_3'].extend([cubic_a[5]])
        results['Cubic_a_oilt_3'].extend([cubic_a[6]])
        results['Cubic_a_fahrt_3'].extend([cubic_a[7]])
        results['Cubic_a_wind_3'].extend([cubic_a[8]])
        results['Cubic_a_clock_3'].extend([cubic_a[9]])
        results['Cubic_b_speed_3'].extend([cubic_b[0]])
        results['Cubic_b_oilp_3'].extend([cubic_b[1]])
        results['Cubic_b_cool_3'].extend([cubic_b[2]])
        results['Cubic_b_boost_3'].extend([cubic_b[3]])
        results['Cubic_b_temp1_3'].extend([cubic_b[4]])
        results['Cubic_b_temp2_3'].extend([cubic_b[5]])
        results['Cubic_b_oilt_3'].extend([cubic_b[6]])
        results['Cubic_b_fahrt_3'].extend([cubic_b[7]])
        results['Cubic_b_wind_3'].extend([cubic_b[8]])
        results['Cubic_b_clock_3'].extend([cubic_b[9]])
        results['Cubic_c_speed_3'].extend([cubic_c[0]])
        results['Cubic_c_oilp_3'].extend([cubic_c[1]])
        results['Cubic_c_cool_3'].extend([cubic_c[2]])
        results['Cubic_c_boost_3'].extend([cubic_c[3]])
        results['Cubic_c_temp1_3'].extend([cubic_c[4]])
        results['Cubic_c_temp2_3'].extend([cubic_c[5]])
        results['Cubic_c_oilt_3'].extend([cubic_c[6]])
        results['Cubic_c_fahrt_3'].extend([cubic_c[7]])
        results['Cubic_c_wind_3'].extend([cubic_c[8]])
        results['Cubic_c_clock_3'].extend([cubic_c[9]])
        results['Cubic_d_3'].extend([cubic_d])
        results['Cubic_MSE_3'].extend([cubic_mse])
        # import test data
        test_Time = test_data['Time'].values  # Time
        test_SpeedKmHr = test_data['SpeedKmHr'].values  # 63
        test_Wind = test_data['Wind'].values
        test_Wind_Speed = test_data['Wind_Speed'].values  # 16
        # 32
        test_Engine1_engine_speed = test_data['Engine1_engine_speed'].values
        # 69
        test_Engine1_lube_oil_pressure = test_data['Engine1_lube_oil_pressure'].values
        # 28
        test_Engine1_coolant_temperature = test_data['Engine1_coolant_temperature'].values
        # 41
        test_Engine1_boost_pressure = test_data['Engine1_boost_pressure'].values
        test_Engine1_engine_load = test_data['Engine1_engine_load'].values  # 5
        # 12
        test_Engine1_exhaust_temperature1 = test_data['Engine1_exhaust_temperature1'].values
        # 29
        test_Engine1_exhaust_temperature2 = test_data['Engine1_exhaust_temperature2'].values
        # 7
        test_Engine1_fuel_consumption = test_data['Engine1_fuel_consumption'].values
        # 13
        test_Engine1_lube_oil_temperature = test_data['Engine1_lube_oil_temperature'].values
        # 8
        test_Engine2_engine_speed = test_data['Engine2_engine_speed'].values
        # 42
        test_Engine2_lube_oil_pressure = test_data['Engine2_lube_oil_pressure'].values
        # 51
        test_Engine2_coolant_temperature = test_data['Engine2_coolant_temperature'].values
        # 1
        test_Engine2_boost_pressure = test_data['Engine2_boost_pressure'].values
        test_Engine2_engine_load = test_data['Engine2_engine_load'].values  # 9
        # 26
        test_Engine2_exhaust_temperature1 = test_data['Engine2_exhaust_temperature1'].values
        # 61
        test_Engine2_exhaust_temperature2 = test_data['Engine2_exhaust_temperature2'].values
        # 52
        test_Engine2_fuel_consumption = test_data['Engine2_fuel_consumption'].values
        # 22
        test_Engine2_lube_oil_temperature = test_data['Engine2_lube_oil_temperature'].values
        # 14
        test_Engine3_engine_speed = test_data['Engine3_engine_speed'].values
        # 56
        test_Engine3_lube_oil_pressure = test_data['Engine3_lube_oil_pressure'].values
        # 57
        test_Engine3_coolant_temperature = test_data['Engine3_coolant_temperature'].values
        # 33
        test_Engine3_boost_pressure = test_data['Engine3_boost_pressure'].values
        # 30
        test_Engine3_engine_load = test_data['Engine3_engine_load'].values
        # 20
        test_Engine3_exhaust_temperature1 = test_data['Engine3_exhaust_temperature1'].values
        # 15
        test_Engine3_exhaust_temperature2 = test_data['Engine3_exhaust_temperature2'].values
        # 62
        test_Engine3_fuel_consumption = test_data['Engine3_fuel_consumption'].values
        # 53
        test_Engine3_lube_oil_temperature = test_data['Engine3_lube_oil_temperature'].values
        test_Clock = test_data['Clock'].values

        # Get the current iteration's values for Model_Linear_1
        linear_a_speed_1 = results['Linear_a_speed_1'][i]
        linear_a_oilp_1 = results['Linear_a_oilp_1'][i]
        linear_a_cool_1 = results['Linear_a_cool_1'][i]
        linear_a_boost_1 = results['Linear_a_boost_1'][i]
        linear_a_temp1_1 = results['Linear_a_temp1_1'][i]
        linear_a_temp2_1 = results['Linear_a_temp2_1'][i]
        linear_a_oilt_1 = results['Linear_a_oilt_1'][i]
        linear_a_wind_1 = results['Linear_a_wind_1'][i]
        linear_a_fahrt_1 = results['Linear_a_fahrt_1'][i]
        linear_a_clock_1 = results['Linear_a_clock_1'][i]
        linear_b_1 = results['Linear_b_1'][i]

        # Calculate Model_Linear_1 using the current iteration's values
        Model_Linear_1 = (
            linear_a_speed_1 * test_Engine1_engine_speed +
            linear_a_oilp_1 * test_Engine1_lube_oil_pressure +
            linear_a_cool_1 * test_Engine1_coolant_temperature +
            linear_a_boost_1 * test_Engine1_boost_pressure +
            linear_a_temp1_1 * test_Engine1_exhaust_temperature1 +
            linear_a_temp2_1 * test_Engine1_exhaust_temperature2 +
            linear_a_oilt_1 * test_Engine1_lube_oil_temperature +
            linear_a_wind_1 * test_Wind +
            linear_a_fahrt_1 * test_SpeedKmHr +
            linear_a_clock_1 * test_Clock +
            linear_b_1
        )

        # Get the current iteration's values for Model_Sen_1
        sen_a_speed_1 = results['Sen_a_speed_1'][i]
        sen_a_oilp_1 = results['Sen_a_oilp_1'][i]
        sen_a_cool_1 = results['Sen_a_cool_1'][i]
        sen_a_boost_1 = results['Sen_a_boost_1'][i]
        sen_a_temp1_1 = results['Sen_a_temp1_1'][i]
        sen_a_temp2_1 = results['Sen_a_temp2_1'][i]
        sen_a_oilt_1 = results['Sen_a_oilt_1'][i]
        sen_a_wind_1 = results['Sen_a_wind_1'][i]
        sen_a_fahrt_1 = results['Sen_a_fahrt_1'][i]
        sen_a_clock_1 = results['Sen_a_clock_1'][i]
        sen_b_1 = results['Sen_b_1'][i]

        # Calculate Model_Sen_1 using the current iteration's values
        Model_Sen_1 = (
            sen_a_speed_1 * test_Engine1_engine_speed +
            sen_a_oilp_1 * test_Engine1_lube_oil_pressure +
            sen_a_cool_1 * test_Engine1_coolant_temperature +
            sen_a_boost_1 * test_Engine1_boost_pressure +
            sen_a_temp1_1 * test_Engine1_exhaust_temperature1 +
            sen_a_temp2_1 * test_Engine1_exhaust_temperature2 +
            sen_a_oilt_1 * test_Engine1_lube_oil_temperature +
            sen_a_wind_1 * test_Wind +
            sen_a_fahrt_1 * test_SpeedKmHr +
            sen_a_clock_1 * test_Clock +
            sen_b_1
        )
        print('sen_a_speed_1:', sen_a_speed_1)
        print('sen_a_clock_1:', sen_a_clock_1)
        print(test_Engine1_engine_speed)

        # Get the current iteration's values for Model_Parabolic_1
        parabolic_a_speed_1 = results['Parabolic_a_speed_1'][i]
        parabolic_b_speed_1 = results['Parabolic_b_speed_1'][i]
        parabolic_a_oilp_1 = results['Parabolic_a_oilp_1'][i]
        parabolic_b_oilp_1 = results['Parabolic_b_oilp_1'][i]
        parabolic_a_cool_1 = results['Parabolic_a_cool_1'][i]
        parabolic_b_cool_1 = results['Parabolic_b_cool_1'][i]
        parabolic_a_boost_1 = results['Parabolic_a_boost_1'][i]
        parabolic_b_boost_1 = results['Parabolic_b_boost_1'][i]
        parabolic_a_temp1_1 = results['Parabolic_a_temp1_1'][i]
        parabolic_b_temp1_1 = results['Parabolic_b_temp1_1'][i]
        parabolic_a_temp2_1 = results['Parabolic_a_temp2_1'][i]
        parabolic_b_temp2_1 = results['Parabolic_b_temp2_1'][i]
        parabolic_a_oilt_1 = results['Parabolic_a_oilt_1'][i]
        parabolic_b_oilt_1 = results['Parabolic_b_oilt_1'][i]
        parabolic_a_wind_1 = results['Parabolic_a_wind_1'][i]
        parabolic_b_wind_1 = results['Parabolic_b_wind_1'][i]
        parabolic_a_fahrt_1 = results['Parabolic_a_fahrt_1'][i]
        parabolic_b_fahrt_1 = results['Parabolic_b_fahrt_1'][i]
        parabolic_a_clock_1 = results['Parabolic_a_clock_1'][i]
        parabolic_b_clock_1 = results['Parabolic_b_clock_1'][i]
        parabolic_c_1 = results['Parabolic_c_1'][i]

        # Calculate Model_Parabolic_1 using the current iteration's values
        Model_Parabolic_1 = (
            parabolic_a_speed_1 * test_Engine1_engine_speed ** 2 +
            parabolic_b_speed_1 * test_Engine1_engine_speed +
            parabolic_a_oilp_1 * test_Engine1_lube_oil_pressure ** 2 +
            parabolic_b_oilp_1 * test_Engine1_lube_oil_pressure +
            parabolic_a_cool_1 * test_Engine1_coolant_temperature ** 2 +
            parabolic_b_cool_1 * test_Engine1_coolant_temperature +
            parabolic_a_boost_1 * test_Engine1_boost_pressure ** 2 +
            parabolic_b_boost_1 * test_Engine1_boost_pressure +
            parabolic_a_temp1_1 * test_Engine1_exhaust_temperature1 ** 2 +
            parabolic_b_temp1_1 * test_Engine1_exhaust_temperature1 +
            parabolic_a_temp2_1 * test_Engine1_exhaust_temperature2 ** 2 +
            parabolic_b_temp2_1 * test_Engine1_exhaust_temperature2 +
            parabolic_a_oilt_1 * test_Engine1_lube_oil_temperature ** 2 +
            parabolic_b_oilt_1 * test_Engine1_lube_oil_temperature +
            parabolic_a_wind_1 * test_Wind ** 2 +
            parabolic_b_wind_1 * test_Wind +
            parabolic_a_fahrt_1 * test_SpeedKmHr ** 2 +
            parabolic_b_fahrt_1 * test_SpeedKmHr +
            parabolic_a_clock_1 * test_Clock ** 2 +
            parabolic_b_clock_1 * test_Clock +
            parabolic_c_1
        )

        # Get the current iteration's values for Model_Cubic_1
        cubic_a_speed_1 = results['Cubic_a_speed_1'][i]
        cubic_b_speed_1 = results['Cubic_b_speed_1'][i]
        cubic_c_speed_1 = results['Cubic_c_speed_1'][i]
        cubic_a_oilp_1 = results['Cubic_a_oilp_1'][i]
        cubic_b_oilp_1 = results['Cubic_b_oilp_1'][i]
        cubic_c_oilp_1 = results['Cubic_c_oilp_1'][i]
        cubic_a_cool_1 = results['Cubic_a_cool_1'][i]
        cubic_b_cool_1 = results['Cubic_b_cool_1'][i]
        cubic_c_cool_1 = results['Cubic_c_cool_1'][i]
        cubic_a_boost_1 = results['Cubic_a_boost_1'][i]
        cubic_b_boost_1 = results['Cubic_b_boost_1'][i]
        cubic_c_boost_1 = results['Cubic_c_boost_1'][i]
        cubic_a_temp1_1 = results['Cubic_a_temp1_1'][i]
        cubic_b_temp1_1 = results['Cubic_b_temp1_1'][i]
        cubic_c_temp1_1 = results['Cubic_c_temp1_1'][i]
        cubic_a_temp2_1 = results['Cubic_a_temp2_1'][i]
        cubic_b_temp2_1 = results['Cubic_b_temp2_1'][i]
        cubic_c_temp2_1 = results['Cubic_c_temp2_1'][i]
        cubic_a_oilt_1 = results['Cubic_a_oilt_1'][i]
        cubic_b_oilt_1 = results['Cubic_b_oilt_1'][i]
        cubic_c_oilt_1 = results['Cubic_c_oilt_1'][i]
        cubic_a_wind_1 = results['Cubic_a_wind_1'][i]
        cubic_b_wind_1 = results['Cubic_b_wind_1'][i]
        cubic_c_wind_1 = results['Cubic_c_wind_1'][i]
        cubic_a_fahrt_1 = results['Cubic_a_fahrt_1'][i]
        cubic_b_fahrt_1 = results['Cubic_b_fahrt_1'][i]
        cubic_c_fahrt_1 = results['Cubic_c_fahrt_1'][i]
        cubic_a_clock_1 = results['Cubic_a_clock_1'][i]
        cubic_b_clock_1 = results['Cubic_b_clock_1'][i]
        cubic_c_clock_1 = results['Cubic_c_clock_1'][i]
        cubic_d_1 = results['Cubic_d_1'][i]

        # Calculate Model_Cubic_1 for the current iteration
        Model_Cubic_1 = (
            cubic_a_speed_1 * test_Engine1_engine_speed ** 3 +
            cubic_b_speed_1 * test_Engine1_engine_speed ** 2 +
            cubic_c_speed_1 * test_Engine1_engine_speed +
            cubic_a_oilp_1 * test_Engine1_lube_oil_pressure ** 3 +
            cubic_b_oilp_1 * test_Engine1_lube_oil_pressure ** 2 +
            cubic_c_oilp_1 * test_Engine1_lube_oil_pressure +
            cubic_a_cool_1 * test_Engine1_coolant_temperature ** 3 +
            cubic_b_cool_1 * test_Engine1_coolant_temperature ** 2 +
            cubic_c_cool_1 * test_Engine1_coolant_temperature +
            cubic_a_boost_1 * test_Engine1_boost_pressure ** 3 +
            cubic_b_boost_1 * test_Engine1_boost_pressure ** 2 +
            cubic_c_boost_1 * test_Engine1_boost_pressure +
            cubic_a_temp1_1 * test_Engine1_exhaust_temperature1 ** 3 +
            cubic_b_temp1_1 * test_Engine1_exhaust_temperature1 ** 2 +
            cubic_c_temp1_1 * test_Engine1_exhaust_temperature1 +
            cubic_a_temp2_1 * test_Engine1_exhaust_temperature2 ** 3 +
            cubic_b_temp2_1 * test_Engine1_exhaust_temperature2 ** 2 +
            cubic_c_temp2_1 * test_Engine1_exhaust_temperature2 +
            cubic_a_oilt_1 * test_Engine1_lube_oil_temperature ** 3 +
            cubic_b_oilt_1 * test_Engine1_lube_oil_temperature ** 2 +
            cubic_c_oilt_1 * test_Engine1_lube_oil_temperature +
            cubic_a_wind_1 * test_Wind ** 3 +
            cubic_b_wind_1 * test_Wind ** 2 +
            cubic_c_wind_1 * test_Wind +
            cubic_a_fahrt_1 * test_SpeedKmHr ** 3 +
            cubic_b_fahrt_1 * test_SpeedKmHr ** 2 +
            cubic_c_fahrt_1 * test_SpeedKmHr +
            cubic_a_clock_1 * test_Clock ** 3 +
            cubic_b_clock_1 * test_Clock ** 2 +
            cubic_c_clock_1 * test_Clock +
            cubic_d_1
        )

        # Model for Engine 2
        # Get the current iteration's values for Model_Linear_2
        linear_a_speed_2 = results['Linear_a_speed_2'][i]
        linear_a_oilp_2 = results['Linear_a_oilp_2'][i]
        linear_a_cool_2 = results['Linear_a_cool_2'][i]
        linear_a_boost_2 = results['Linear_a_boost_2'][i]
        linear_a_temp1_2 = results['Linear_a_temp1_2'][i]
        linear_a_temp2_2 = results['Linear_a_temp2_2'][i]
        linear_a_oilt_2 = results['Linear_a_oilt_2'][i]
        linear_a_wind_2 = results['Linear_a_wind_2'][i]
        linear_a_fahrt_2 = results['Linear_a_fahrt_2'][i]
        linear_a_clock_2 = results['Linear_a_clock_2'][i]
        linear_b_2 = results['Linear_b_2'][i]

        # Calculate Model_Linear_2 using the current iteration's values
        Model_Linear_2 = (
            linear_a_speed_2 * test_Engine2_engine_speed +
            linear_a_oilp_2 * test_Engine2_lube_oil_pressure +
            linear_a_cool_2 * test_Engine2_coolant_temperature +
            linear_a_boost_2 * test_Engine2_boost_pressure +
            linear_a_temp1_2 * test_Engine2_exhaust_temperature1 +
            linear_a_temp2_2 * test_Engine2_exhaust_temperature2 +
            linear_a_oilt_2 * test_Engine2_lube_oil_temperature +
            linear_a_wind_2 * test_Wind +
            linear_a_fahrt_2 * test_SpeedKmHr +
            linear_a_clock_2 * test_Clock +
            linear_b_2
        )

        # Get the current iteration's values for Model_Sen_2
        sen_a_speed_2 = results['Sen_a_speed_2'][i]
        sen_a_oilp_2 = results['Sen_a_oilp_2'][i]
        sen_a_cool_2 = results['Sen_a_cool_2'][i]
        sen_a_boost_2 = results['Sen_a_boost_2'][i]
        sen_a_temp1_2 = results['Sen_a_temp1_2'][i]
        sen_a_temp2_2 = results['Sen_a_temp2_2'][i]
        sen_a_oilt_2 = results['Sen_a_oilt_2'][i]
        sen_a_wind_2 = results['Sen_a_wind_2'][i]
        sen_a_fahrt_2 = results['Sen_a_fahrt_2'][i]
        sen_a_clock_2 = results['Sen_a_clock_2'][i]
        sen_b_2 = results['Sen_b_2'][i]

        # Calculate Model_Sen_2 using the current iteration's values
        Model_Sen_2 = (
            sen_a_speed_2 * test_Engine2_engine_speed +
            sen_a_oilp_2 * test_Engine2_lube_oil_pressure +
            sen_a_cool_2 * test_Engine2_coolant_temperature +
            sen_a_boost_2 * test_Engine2_boost_pressure +
            sen_a_temp1_2 * test_Engine2_exhaust_temperature1 +
            sen_a_temp2_2 * test_Engine2_exhaust_temperature2 +
            sen_a_oilt_2 * test_Engine2_lube_oil_temperature +
            sen_a_wind_2 * test_Wind +
            sen_a_fahrt_2 * test_SpeedKmHr +
            sen_a_clock_2 * test_Clock +
            sen_b_2
        )

        # Get the current iteration's values for Model_Parabolic_2
        parabolic_a_speed_2 = results['Parabolic_a_speed_2'][i]
        parabolic_b_speed_2 = results['Parabolic_b_speed_2'][i]
        parabolic_a_oilp_2 = results['Parabolic_a_oilp_2'][i]
        parabolic_b_oilp_2 = results['Parabolic_b_oilp_2'][i]
        parabolic_a_cool_2 = results['Parabolic_a_cool_2'][i]
        parabolic_b_cool_2 = results['Parabolic_b_cool_2'][i]
        parabolic_a_boost_2 = results['Parabolic_a_boost_2'][i]
        parabolic_b_boost_2 = results['Parabolic_b_boost_2'][i]
        parabolic_a_temp1_2 = results['Parabolic_a_temp1_2'][i]
        parabolic_b_temp1_2 = results['Parabolic_b_temp1_2'][i]
        parabolic_a_temp2_2 = results['Parabolic_a_temp2_2'][i]
        parabolic_b_temp2_2 = results['Parabolic_b_temp2_2'][i]
        parabolic_a_oilt_2 = results['Parabolic_a_oilt_2'][i]
        parabolic_b_oilt_2 = results['Parabolic_b_oilt_2'][i]
        parabolic_a_wind_2 = results['Parabolic_a_wind_2'][i]
        parabolic_b_wind_2 = results['Parabolic_b_wind_2'][i]
        parabolic_a_fahrt_2 = results['Parabolic_a_fahrt_2'][i]
        parabolic_b_fahrt_2 = results['Parabolic_b_fahrt_2'][i]
        parabolic_a_clock_2 = results['Parabolic_a_clock_2'][i]
        parabolic_b_clock_2 = results['Parabolic_b_clock_2'][i]
        parabolic_c_2 = results['Parabolic_c_2'][i]

        # Calculate Model_Parabolic_2 using the current iteration's values
        Model_Parabolic_2 = (
            parabolic_a_speed_2 * test_Engine2_engine_speed ** 2 +
            parabolic_b_speed_2 * test_Engine2_engine_speed +
            parabolic_a_oilp_2 * test_Engine2_lube_oil_pressure ** 2 +
            parabolic_b_oilp_2 * test_Engine2_lube_oil_pressure +
            parabolic_a_cool_2 * test_Engine2_coolant_temperature ** 2 +
            parabolic_b_cool_2 * test_Engine2_coolant_temperature +
            parabolic_a_boost_2 * test_Engine2_boost_pressure ** 2 +
            parabolic_b_boost_2 * test_Engine2_boost_pressure +
            parabolic_a_temp1_2 * test_Engine2_exhaust_temperature1 ** 2 +
            parabolic_b_temp1_2 * test_Engine2_exhaust_temperature1 +
            parabolic_a_temp2_2 * test_Engine2_exhaust_temperature2 ** 2 +
            parabolic_b_temp2_2 * test_Engine2_exhaust_temperature2 +
            parabolic_a_oilt_2 * test_Engine2_lube_oil_temperature ** 2 +
            parabolic_b_oilt_2 * test_Engine2_lube_oil_temperature +
            parabolic_a_wind_2 * test_Wind ** 2 +
            parabolic_b_wind_2 * test_Wind +
            parabolic_a_fahrt_2 * test_SpeedKmHr ** 2 +
            parabolic_b_fahrt_2 * test_SpeedKmHr +
            parabolic_a_clock_2 * test_Clock ** 2 +
            parabolic_b_clock_2 * test_Clock +
            parabolic_c_2
        )

        # Get the current iteration's values for Model_Cubic_2
        cubic_a_speed_2 = results['Cubic_a_speed_2'][i]
        cubic_b_speed_2 = results['Cubic_b_speed_2'][i]
        cubic_c_speed_2 = results['Cubic_c_speed_2'][i]
        cubic_a_oilp_2 = results['Cubic_a_oilp_2'][i]
        cubic_b_oilp_2 = results['Cubic_b_oilp_2'][i]
        cubic_c_oilp_2 = results['Cubic_c_oilp_2'][i]
        cubic_a_cool_2 = results['Cubic_a_cool_2'][i]
        cubic_b_cool_2 = results['Cubic_b_cool_2'][i]
        cubic_c_cool_2 = results['Cubic_c_cool_2'][i]
        cubic_a_boost_2 = results['Cubic_a_boost_2'][i]
        cubic_b_boost_2 = results['Cubic_b_boost_2'][i]
        cubic_c_boost_2 = results['Cubic_c_boost_2'][i]
        cubic_a_temp1_2 = results['Cubic_a_temp1_2'][i]
        cubic_b_temp1_2 = results['Cubic_b_temp1_2'][i]
        cubic_c_temp1_2 = results['Cubic_c_temp1_2'][i]
        cubic_a_temp2_2 = results['Cubic_a_temp2_2'][i]
        cubic_b_temp2_2 = results['Cubic_b_temp2_2'][i]
        cubic_c_temp2_2 = results['Cubic_c_temp2_2'][i]
        cubic_a_oilt_2 = results['Cubic_a_oilt_2'][i]
        cubic_b_oilt_2 = results['Cubic_b_oilt_2'][i]
        cubic_c_oilt_2 = results['Cubic_c_oilt_2'][i]
        cubic_a_wind_2 = results['Cubic_a_wind_2'][i]
        cubic_b_wind_2 = results['Cubic_b_wind_2'][i]
        cubic_c_wind_2 = results['Cubic_c_wind_2'][i]
        cubic_a_fahrt_2 = results['Cubic_a_fahrt_2'][i]
        cubic_b_fahrt_2 = results['Cubic_b_fahrt_2'][i]
        cubic_c_fahrt_2 = results['Cubic_c_fahrt_2'][i]
        cubic_a_clock_2 = results['Cubic_a_clock_2'][i]
        cubic_b_clock_2 = results['Cubic_b_clock_2'][i]
        cubic_c_clock_2 = results['Cubic_c_clock_2'][i]
        cubic_d_2 = results['Cubic_d_2'][i]

        # Calculate Model_Cubic_2 for the current iteration
        Model_Cubic_2 = (
            cubic_a_speed_2 * test_Engine2_engine_speed ** 3 +
            cubic_b_speed_2 * test_Engine2_engine_speed ** 2 +
            cubic_c_speed_2 * test_Engine2_engine_speed +
            cubic_a_oilp_2 * test_Engine2_lube_oil_pressure ** 3 +
            cubic_b_oilp_2 * test_Engine2_lube_oil_pressure ** 2 +
            cubic_c_oilp_2 * test_Engine2_lube_oil_pressure +
            cubic_a_cool_2 * test_Engine2_coolant_temperature ** 3 +
            cubic_b_cool_2 * test_Engine2_coolant_temperature ** 2 +
            cubic_c_cool_2 * test_Engine2_coolant_temperature +
            cubic_a_boost_2 * test_Engine2_boost_pressure ** 3 +
            cubic_b_boost_2 * test_Engine2_boost_pressure ** 2 +
            cubic_c_boost_2 * test_Engine2_boost_pressure +
            cubic_a_temp1_2 * test_Engine2_exhaust_temperature1 ** 3 +
            cubic_b_temp1_2 * test_Engine2_exhaust_temperature1 ** 2 +
            cubic_c_temp1_2 * test_Engine2_exhaust_temperature1 +
            cubic_a_temp2_2 * test_Engine2_exhaust_temperature2 ** 3 +
            cubic_b_temp2_2 * test_Engine2_exhaust_temperature2 ** 2 +
            cubic_c_temp2_2 * test_Engine2_exhaust_temperature2 +
            cubic_a_oilt_2 * test_Engine2_lube_oil_temperature ** 3 +
            cubic_b_oilt_2 * test_Engine2_lube_oil_temperature ** 2 +
            cubic_c_oilt_2 * test_Engine2_lube_oil_temperature +
            cubic_a_wind_2 * test_Wind ** 3 +
            cubic_b_wind_2 * test_Wind ** 2 +
            cubic_c_wind_2 * test_Wind +
            cubic_a_fahrt_2 * test_SpeedKmHr ** 3 +
            cubic_b_fahrt_2 * test_SpeedKmHr ** 2 +
            cubic_c_fahrt_2 * test_SpeedKmHr +
            cubic_a_clock_2 * test_Clock ** 3 +
            cubic_b_clock_2 * test_Clock ** 2 +
            cubic_c_clock_2 * test_Clock +
            cubic_d_2
        )

        # Model for Engine 3
        # Get the current iteration's values for Model_Linear_3
        linear_a_speed_3 = results['Linear_a_speed_3'][i]
        linear_a_oilp_3 = results['Linear_a_oilp_3'][i]
        linear_a_cool_3 = results['Linear_a_cool_3'][i]
        linear_a_boost_3 = results['Linear_a_boost_3'][i]
        linear_a_temp1_3 = results['Linear_a_temp1_3'][i]
        linear_a_temp2_3 = results['Linear_a_temp2_3'][i]
        linear_a_oilt_3 = results['Linear_a_oilt_3'][i]
        linear_a_wind_3 = results['Linear_a_wind_3'][i]
        linear_a_fahrt_3 = results['Linear_a_fahrt_3'][i]
        linear_a_clock_3 = results['Linear_a_clock_3'][i]
        linear_b_3 = results['Linear_b_3'][i]

        # Calculate Model_Linear_3 using the current iteration's values
        Model_Linear_3 = (
            linear_a_speed_3 * test_Engine3_engine_speed +
            linear_a_oilp_3 * test_Engine3_lube_oil_pressure +
            linear_a_cool_3 * test_Engine3_coolant_temperature +
            linear_a_boost_3 * test_Engine3_boost_pressure +
            linear_a_temp1_3 * test_Engine3_exhaust_temperature1 +
            linear_a_temp2_3 * test_Engine3_exhaust_temperature2 +
            linear_a_oilt_3 * test_Engine3_lube_oil_temperature +
            linear_a_wind_3 * test_Wind +
            linear_a_fahrt_3 * test_SpeedKmHr +
            linear_a_clock_3 * test_Clock +
            linear_b_3
        )

        # Get the current iteration's values for Model_Sen_3
        sen_a_speed_3 = results['Sen_a_speed_3'][i]
        sen_a_oilp_3 = results['Sen_a_oilp_3'][i]
        sen_a_cool_3 = results['Sen_a_cool_3'][i]
        sen_a_boost_3 = results['Sen_a_boost_3'][i]
        sen_a_temp1_3 = results['Sen_a_temp1_3'][i]
        sen_a_temp2_3 = results['Sen_a_temp2_3'][i]
        sen_a_oilt_3 = results['Sen_a_oilt_3'][i]
        sen_a_wind_3 = results['Sen_a_wind_3'][i]
        sen_a_fahrt_3 = results['Sen_a_fahrt_3'][i]
        sen_a_clock_3 = results['Sen_a_clock_3'][i]
        sen_b_3 = results['Sen_b_3'][i]

        # Calculate Model_Sen_3 using the current iteration's values
        Model_Sen_3 = (
            sen_a_speed_3 * test_Engine3_engine_speed +
            sen_a_oilp_3 * test_Engine3_lube_oil_pressure +
            sen_a_cool_3 * test_Engine3_coolant_temperature +
            sen_a_boost_3 * test_Engine3_boost_pressure +
            sen_a_temp1_3 * test_Engine3_exhaust_temperature1 +
            sen_a_temp2_3 * test_Engine3_exhaust_temperature2 +
            sen_a_oilt_3 * test_Engine3_lube_oil_temperature +
            sen_a_wind_3 * test_Wind +
            sen_a_fahrt_3 * test_SpeedKmHr +
            sen_a_clock_3 * test_Clock +
            sen_b_3
        )

        # Get the current iteration's values for Model_Parabolic_3
        parabolic_a_speed_3 = results['Parabolic_a_speed_3'][i]
        parabolic_b_speed_3 = results['Parabolic_b_speed_3'][i]
        parabolic_a_oilp_3 = results['Parabolic_a_oilp_3'][i]
        parabolic_b_oilp_3 = results['Parabolic_b_oilp_3'][i]
        parabolic_a_cool_3 = results['Parabolic_a_cool_3'][i]
        parabolic_b_cool_3 = results['Parabolic_b_cool_3'][i]
        parabolic_a_boost_3 = results['Parabolic_a_boost_3'][i]
        parabolic_b_boost_3 = results['Parabolic_b_boost_3'][i]
        parabolic_a_temp1_3 = results['Parabolic_a_temp1_3'][i]
        parabolic_b_temp1_3 = results['Parabolic_b_temp1_3'][i]
        parabolic_a_temp2_3 = results['Parabolic_a_temp2_3'][i]
        parabolic_b_temp2_3 = results['Parabolic_b_temp2_3'][i]
        parabolic_a_oilt_3 = results['Parabolic_a_oilt_3'][i]
        parabolic_b_oilt_3 = results['Parabolic_b_oilt_3'][i]
        parabolic_a_wind_3 = results['Parabolic_a_wind_3'][i]
        parabolic_b_wind_3 = results['Parabolic_b_wind_3'][i]
        parabolic_a_fahrt_3 = results['Parabolic_a_fahrt_3'][i]
        parabolic_b_fahrt_3 = results['Parabolic_b_fahrt_3'][i]
        parabolic_a_clock_3 = results['Parabolic_a_clock_3'][i]
        parabolic_b_clock_3 = results['Parabolic_b_clock_3'][i]
        parabolic_c_3 = results['Parabolic_c_3'][i]

        # Calculate Model_Parabolic_3 using the current iteration's values
        Model_Parabolic_3 = (
            parabolic_a_speed_3 * test_Engine3_engine_speed ** 3 +
            parabolic_b_speed_3 * test_Engine3_engine_speed +
            parabolic_a_oilp_3 * test_Engine3_lube_oil_pressure ** 3 +
            parabolic_b_oilp_3 * test_Engine3_lube_oil_pressure +
            parabolic_a_cool_3 * test_Engine3_coolant_temperature ** 3 +
            parabolic_b_cool_3 * test_Engine3_coolant_temperature +
            parabolic_a_boost_3 * test_Engine3_boost_pressure ** 3 +
            parabolic_b_boost_3 * test_Engine3_boost_pressure +
            parabolic_a_temp1_3 * test_Engine3_exhaust_temperature1 ** 3 +
            parabolic_b_temp1_3 * test_Engine3_exhaust_temperature1 +
            parabolic_a_temp2_3 * test_Engine3_exhaust_temperature2 ** 3 +
            parabolic_b_temp2_3 * test_Engine3_exhaust_temperature2 +
            parabolic_a_oilt_3 * test_Engine3_lube_oil_temperature ** 3 +
            parabolic_b_oilt_3 * test_Engine3_lube_oil_temperature +
            parabolic_a_wind_3 * test_Wind ** 3 +
            parabolic_b_wind_3 * test_Wind +
            parabolic_a_fahrt_3 * test_SpeedKmHr ** 3 +
            parabolic_b_fahrt_3 * test_SpeedKmHr +
            parabolic_a_clock_3 * test_Clock ** 3 +
            parabolic_b_clock_3 * test_Clock +
            parabolic_c_3
        )

        # Get the current iteration's values for Model_Cubic_3
        cubic_a_speed_3 = results['Cubic_a_speed_3'][i]
        cubic_b_speed_3 = results['Cubic_b_speed_3'][i]
        cubic_c_speed_3 = results['Cubic_c_speed_3'][i]
        cubic_a_oilp_3 = results['Cubic_a_oilp_3'][i]
        cubic_b_oilp_3 = results['Cubic_b_oilp_3'][i]
        cubic_c_oilp_3 = results['Cubic_c_oilp_3'][i]
        cubic_a_cool_3 = results['Cubic_a_cool_3'][i]
        cubic_b_cool_3 = results['Cubic_b_cool_3'][i]
        cubic_c_cool_3 = results['Cubic_c_cool_3'][i]
        cubic_a_boost_3 = results['Cubic_a_boost_3'][i]
        cubic_b_boost_3 = results['Cubic_b_boost_3'][i]
        cubic_c_boost_3 = results['Cubic_c_boost_3'][i]
        cubic_a_temp1_3 = results['Cubic_a_temp1_3'][i]
        cubic_b_temp1_3 = results['Cubic_b_temp1_3'][i]
        cubic_c_temp1_3 = results['Cubic_c_temp1_3'][i]
        cubic_a_temp2_3 = results['Cubic_a_temp2_3'][i]
        cubic_b_temp2_3 = results['Cubic_b_temp2_3'][i]
        cubic_c_temp2_3 = results['Cubic_c_temp2_3'][i]
        cubic_a_oilt_3 = results['Cubic_a_oilt_3'][i]
        cubic_b_oilt_3 = results['Cubic_b_oilt_3'][i]
        cubic_c_oilt_3 = results['Cubic_c_oilt_3'][i]
        cubic_a_wind_3 = results['Cubic_a_wind_3'][i]
        cubic_b_wind_3 = results['Cubic_b_wind_3'][i]
        cubic_c_wind_3 = results['Cubic_c_wind_3'][i]
        cubic_a_fahrt_3 = results['Cubic_a_fahrt_3'][i]
        cubic_b_fahrt_3 = results['Cubic_b_fahrt_3'][i]
        cubic_c_fahrt_3 = results['Cubic_c_fahrt_3'][i]
        cubic_a_clock_3 = results['Cubic_a_clock_3'][i]
        cubic_b_clock_3 = results['Cubic_b_clock_3'][i]
        cubic_c_clock_3 = results['Cubic_c_clock_3'][i]
        cubic_d_3 = results['Cubic_d_3'][i]

        # Calculate Model_Cubic_3 for the current iteration
        Model_Cubic_3 = (
            cubic_a_speed_3 * test_Engine3_engine_speed ** 3 +
            cubic_b_speed_3 * test_Engine3_engine_speed ** 3 +
            cubic_c_speed_3 * test_Engine3_engine_speed +
            cubic_a_oilp_3 * test_Engine3_lube_oil_pressure ** 3 +
            cubic_b_oilp_3 * test_Engine3_lube_oil_pressure ** 3 +
            cubic_c_oilp_3 * test_Engine3_lube_oil_pressure +
            cubic_a_cool_3 * test_Engine3_coolant_temperature ** 3 +
            cubic_b_cool_3 * test_Engine3_coolant_temperature ** 3 +
            cubic_c_cool_3 * test_Engine3_coolant_temperature +
            cubic_a_boost_3 * test_Engine3_boost_pressure ** 3 +
            cubic_b_boost_3 * test_Engine3_boost_pressure ** 3 +
            cubic_c_boost_3 * test_Engine3_boost_pressure +
            cubic_a_temp1_3 * test_Engine3_exhaust_temperature1 ** 3 +
            cubic_b_temp1_3 * test_Engine3_exhaust_temperature1 ** 3 +
            cubic_c_temp1_3 * test_Engine3_exhaust_temperature1 +
            cubic_a_temp2_3 * test_Engine3_exhaust_temperature2 ** 3 +
            cubic_b_temp2_3 * test_Engine3_exhaust_temperature2 ** 3 +
            cubic_c_temp2_3 * test_Engine3_exhaust_temperature2 +
            cubic_a_oilt_3 * test_Engine3_lube_oil_temperature ** 3 +
            cubic_b_oilt_3 * test_Engine3_lube_oil_temperature ** 3 +
            cubic_c_oilt_3 * test_Engine3_lube_oil_temperature +
            cubic_a_wind_3 * test_Wind ** 3 +
            cubic_b_wind_3 * test_Wind ** 3 +
            cubic_c_wind_3 * test_Wind +
            cubic_a_fahrt_3 * test_SpeedKmHr ** 3 +
            cubic_b_fahrt_3 * test_SpeedKmHr ** 3 +
            cubic_c_fahrt_3 * test_SpeedKmHr +
            cubic_a_clock_3 * test_Clock ** 3 +
            cubic_b_clock_3 * test_Clock ** 3 +
            cubic_c_clock_3 * test_Clock +
            cubic_d_3
        )

        # Caculate MSE
        mse_linear_1 = calculate_mse(Model_Linear_1, test_Engine1_engine_load)
        mse_sen_1 = calculate_mse(Model_Sen_1, test_Engine1_engine_load)
        mse_parabolic_1 = calculate_mse(
            Model_Parabolic_1, test_Engine1_engine_load)
        mse_cubic_1 = calculate_mse(Model_Cubic_1, test_Engine1_engine_load)

        # Append the data frame
        results['Linear_test_MSE_1'].extend([mse_linear_1])
        results['Sen_test_MSE_1'].extend([mse_sen_1])
        results['Parabolic_test_MSE_1'].extend([mse_parabolic_1])
        results['Cubic_test_MSE_1'].extend([mse_cubic_1])

        # Caculate MSE
        mse_linear_2 = calculate_mse(Model_Linear_2, test_Engine2_engine_load)
        mse_sen_2 = calculate_mse(Model_Sen_2, test_Engine2_engine_load)
        mse_parabolic_2 = calculate_mse(
            Model_Parabolic_2, test_Engine2_engine_load)
        mse_cubic_2 = calculate_mse(Model_Cubic_2, test_Engine2_engine_load)

        # Append the data frame
        results['Linear_test_MSE_2'].extend([mse_linear_2])
        results['Sen_test_MSE_2'].extend([mse_sen_2])
        results['Parabolic_test_MSE_2'].extend([mse_parabolic_2])
        results['Cubic_test_MSE_2'].extend([mse_cubic_2])

        # Caculate MSE
        mse_linear_3 = calculate_mse(Model_Linear_3, test_Engine3_engine_load)
        mse_sen_3 = calculate_mse(Model_Sen_3, test_Engine3_engine_load)
        mse_parabolic_3 = calculate_mse(
            Model_Parabolic_3, test_Engine3_engine_load)
        mse_cubic_3 = calculate_mse(Model_Cubic_3, test_Engine3_engine_load)

        # Append the data frame
        results['Linear_test_MSE_3'].extend([mse_linear_3])
        results['Sen_test_MSE_3'].extend([mse_sen_3])
        results['Parabolic_test_MSE_3'].extend([mse_parabolic_3])
        results['Cubic_test_MSE_3'].extend([mse_cubic_3])

        # Example: Print the file path
        print("Performing black box code on:", filename)

        # Add a separator between subfolders for clarity
        print("--------------------")

        i += 1

    # Pad the arrays with None to ensure they have the same maximum length
    pad_max_length_arrays(results) 

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(results)

    # Save the DataFrame as a CSV file
    df.to_csv('./results_experiment1_1.csv', sep=';', index=False)

def get_dataset_files(dataset_dir: str) -> List[str]:
    """Lists the available dataset CSV files"""
    for root, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                full_path = os.path.join(root, filename)
                yield full_path

def display_available_dataset_files(dataset_files: List[str]) -> None:
    print('Available dataset files:\n')
    for filename in dataset_files:
        print(filename)

def main() -> None:
    parser = argparse.ArgumentParser(
        prog='experiment',
        description='runs experiments 🤦‍♂️',
        usage=__doc__)

    parser.add_argument('-f', '--input-folder',
                        help='The input folder containing CSV dataset files.', action='store')
    parser.add_argument('-l', '--list',
                        help=("Lists the available CSV dataset files if the '--input-folder' option is argument."
                              "Use those values with the '--input' argument."), action='store_true')
    parser.add_argument(
        '-i', '--input', help='The path and name to an input CSV dataset file to include.', nargs='*')
    parser.add_argument(
        '-o', '--output', help='The path and name to the resulting experiment report CSV file.', action='store')

    args = vars(parser.parse_args())

    has_input = has_dict_key(args, 'input')
    has_input_folder = has_dict_key(args, 'input_folder')

    if not has_input and not has_input_folder:
        stderr_exit("Invalid arguments: you MUST specify one of '--input' or '--input-folder' argument.")

    if not has_dict_key(args, 'output'):
        args['output'] = './results_experiment1_1.csv'

    if has_input:
        args['dataset_files'] = args['input']
    else:
        args['dataset_files'] = get_dataset_files(args['input_folder'])

    if args['list']:
        if not has_input_folder:
            stderr_exit("Invalid arguments: when specifying the '--list' argument, you MUST also specify '--input-folder' argument.")
        display_available_dataset_files(args['input'])

    else:
        run(args)

if __name__ == '__main__':
    main()
