
from math import sqrt
from typing import List, Callable


data_set = [-5, 10, 15]




def calculate_standard_deviation(dataset: List[float]) -> float:
    sum_of_squares = 0
    mean = calculate_mean(dataset)
    for number in data_set:
        sum_of_squares += ((mean-number)**2)

    return sqrt(sum_of_squares/len(dataset))


def calculate_mean(dataset: List[float]) -> float:
    return sum(dataset)/len(dataset)


def standardize(x: float, dataset: List[float]) -> float:
    mean = calculate_mean(dataset)
    standard_deviation = calculate_standard_deviation(dataset)
    standardized_x = (x-mean)/standard_deviation
    return standardized_x


def normalize(x: float, dataset: List[float]) -> float:
    maximum_value = max(dataset)
    minimum_value = min(dataset)
    return ((x-minimum_value)/(maximum_value-minimum_value))


print("The mean is: ", calculate_mean(data_set))
print("The standard deviation is", calculate_standard_deviation(data_set))

for x in data_set:
    print(standardize(x, data_set))
    print("Nomarlized: ", normalize(x, data_set))
