
from math import sqrt


data_set = [100, 50, 150]


def calculate_standard_deviation(dataset: list) -> float:
    sum_of_squares = 0
    mean = calculate_mean(dataset)
    for number in data_set:
        sum_of_squares += ((mean-number)**2)

    return sqrt(sum_of_squares/len(dataset))


def calculate_mean(dataset: list) -> float:
    return sum(dataset)/len(dataset)


def standardize(x: float, dataset: list) -> float:
    mean = calculate_mean(dataset)
    standard_deviation = calculate_standard_deviation(dataset)
    standardized_x = (x-mean)/standard_deviation
    return standardized_x


print("The mean is: ", calculate_mean(data_set))
print("The standard deviation is", calculate_standard_deviation(data_set))

for x in data_set:
    print(standardize(x, data_set))
