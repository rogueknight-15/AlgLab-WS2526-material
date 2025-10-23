from data_schema import Instance, Solution
from ortools.sat.python import cp_model


def solve(instance: Instance) -> Solution:
    """
    Implement your solver for the problem here!
    """

    """
    maximize sum(x_i * a_i) - sum(y_i * a_i)
    s.t 
        sum(x_i) = 1
        sum(y_i) = 1

    x_i, y_i ∈ {0, 1}
    i ∈ N
    a_i = given list of numbers
    """

    numbers = instance.numbers
    model = cp_model.CpModel()

    # decision variables for x and y
    xs = [model.new_bool_var(f"x{i}") for i in range(len(numbers))]
    ys = [model.new_bool_var(f"y{i}") for i in range(len(numbers))]

    # constraints for decision variables
    # exactly one decision variable can be 1
    model.add_exactly_one(*xs)
    model.add_exactly_one(*ys)

    # maximize the distance between any two numbers in numbers
    # that are seleced via the decision variables x and y
    x_sum = sum(xs[i] * numbers[i] for i in range(len(numbers)))
    y_sum = sum(ys[i] * numbers[i] for i in range(len(numbers)))
    model.maximize(x_sum - y_sum)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    assert status == cp_model.OPTIMAL

    # get the first index for x and y where the decision variable is 1
    x_index = next((i for i, x in enumerate(xs) if solver.value(x) == 1))
    y_index = next((i for i, y in enumerate(ys) if solver.value(y) == 1))

    return Solution(
        number_a=numbers[x_index],
        number_b=numbers[y_index],
        distance=solver.objective_value,
    )
