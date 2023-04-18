import string
import types

import numba
import numpy as np
from numba import typed, types
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

from src.genetic_program.config import FITNESS_FUNCTION
from src.tree.node import Node
from src.tree.tree_util import get_postfix


@numba.njit
def evaluate(row, expression):
    tokens = expression.split()
    stack = typed.List.empty_list(types.float64)
    row = row.astype(np.float32)
    alphabet = list(string.ascii_lowercase)

    for token in tokens:
        if token in ['+', '-', '*', '/', "sqrt"]:
            b = stack.pop()
            a = stack.pop()

            if token == '+':
                result = a + b
            elif token == '*':
                result = a * b
            elif token == '-':
                result = a - b
            elif token == '/':
                if b != 0:
                    result = a / b
                else:
                    result = 0
            else:
                result = np.sqrt(abs(max(a, b)))

            stack.append(result)

        else:
            value: float = float(row[alphabet.index(token)])
            stack.append(abs(value))

    return abs(stack.pop())


def calculate_fitness(tree: Node, x, y):
    expr = get_postfix(tree)

    fitness_evaluation = lambda row, expression: evaluate(row, expression)
    y_pred = np.apply_along_axis(fitness_evaluation, axis=1, arr=x, expression=expr)

    if FITNESS_FUNCTION == 'RMSE':
        tree.fitness = np.sqrt(mean_squared_error(y, y_pred))
    elif FITNESS_FUNCTION == 'MAE':
        tree.fitness = mean_absolute_error(y, y_pred)
    elif FITNESS_FUNCTION == 'MedAE':
        tree.fitness = median_absolute_error(y, y_pred)
