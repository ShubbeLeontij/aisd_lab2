#!/usr/bin/env python3
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings


def sum_of_squared_error(parameterTuple):  # function for genetic algorithm to minimize (sum of squared error)
    warnings.filterwarnings("ignore")
    return numpy.sum((yData - func(xData, *parameterTuple)) ** 2.0)


def generate_initial_parameters():
    parameterBounds = [[min(xData), max(xData)], [min(xData), max(xData)]]
    r = differential_evolution(sum_of_squared_error, parameterBounds, seed=3)
    return r.x


if __name__ == "__main__":
    try:
        with open("result_fast.json", 'r') as file:
            result = json.load(file)["result"]
        functions = [
            lambda x, a, b: a * x,
            lambda x, a, b: a * x**2,
            lambda x, a, b: a * x**1.5,
            lambda x, a, b: a * x * numpy.log(x),
        ]
        for sort, sort_result in result.items():
            for mode, mode_result in sort_result.items():
                xData = numpy.array(list(map(float, mode_result.keys())))
                yData = numpy.array(list(map(float, mode_result.values())))

                min_MSE, i = None, 0
                for func in functions:
                    i += 1
                    genetic_parameters = generate_initial_parameters()
                    fitted_parameters, pcov = curve_fit(func, xData, yData, genetic_parameters)

                    model_predictions = func(xData, *fitted_parameters)
                    abs_error = model_predictions - yData
                    SE = numpy.square(abs_error)  # squared errors
                    MSE = numpy.mean(SE)  # mean squared errors
                    if min_MSE is None or min_MSE > MSE:
                        params, case, min_MSE = fitted_parameters, i, MSE
                    R_squared = 1.0 - (numpy.var(abs_error) / numpy.var(yData))

                if case == 1:
                    func_string = " * N"
                elif case == 2:
                    func_string = " * N^2"
                elif case == 3:
                    func_string = " * N^1,5"
                else:
                    func_string = " * N * ln(N)"
                xModel = numpy.linspace(min(xData), max(xData))
                yModel = functions[case - 1](xModel, *params)
                matplotlib.pyplot.scatter(xData, yData, color="#000000")
                matplotlib.pyplot.plot(xModel, yModel)
                matplotlib.pyplot.title(sort + " sort, " + mode + " array:\n" + str(params[0]) + func_string)# + " + " + str(params[1]))
                matplotlib.pyplot.xlabel("Array size")
                matplotlib.pyplot.ylabel("Time, seconds")
                matplotlib.pyplot.show()
    except AttributeError:
        pass
