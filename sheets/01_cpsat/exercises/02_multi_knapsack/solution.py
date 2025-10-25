import math

from data_schema import Instance, Solution
from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL, CpModel, CpSolver


class MultiKnapsackSolver:
    """
    This class can be used to solve the Multi-Knapsack problem
    (also the standard knapsack problem, if only one capacity is used).

    Attributes:
    - instance (Instance): The multi-knapsack instance
        - items (List[Item]): a list of Item objects representing the items to be packed.
        - capacities (List[int]): a list of integers representing the capacities of the knapsacks.
    - model (CpModel): a CpModel object representing the constraint programming model.
    - solver (CpSolver): a CpSolver object representing the constraint programming solver.
    """

    def __init__(self, instance: Instance, activate_toxic: bool = False):
        """
        Initialize the solver with the given Multi-Knapsack instance.

        Args:
        - instance (Instance): an Instance object representing the Multi-Knapsack instance.
        """
        self.items = instance.items
        self.activate_toxic = activate_toxic
        self.capacities = instance.capacities
        self.model = CpModel()
        self.solver = CpSolver()
        self.solver.parameters.log_search_progress = True

        # precompute values and items for easier access
        self.values = [item.value for item in self.items]
        self.weights = [item.weight for item in self.items]

    def solve(self, timelimit: float = math.inf) -> Solution:
        """
        Solve the Multi-Knapsack instance with the given time limit.

        Args:
        - timelimit (float): time limit in seconds for the cp-sat solver.

        Returns:
        - Solution: a list of lists of Item objects representing the items packed in each knapsack
        """
        # handle given time limit
        if timelimit <= 0.0:
            return Solution(trucks=[])  # empty solution
        if timelimit < math.inf:
            self.solver.parameters.max_time_in_seconds = timelimit

        # decision variable x for each item i and truck j
        x = {
            (i, j): self.model.new_bool_var(f"x_{i}_{j}") for i in range(len(self.items)) for j in range(len(self.capacities))
        }

        # decision variable y for each truck j. 
        # marks if a truck has a toxic item
        y = {
            j: self.model.new_bool_var(f"y_{j}") for j in range(len(self.capacities)) 
        }

        # for all trucks j the sum of weights cannot exceed the trucks capacity
        for j in range(len(self.capacities)):
            self.model.add(sum(self.weights[i] * x[i, j] for i in range(len(self.items)))<= self.capacities[j])

        # for each item i there can only be at most one of it in a truck
        for i in range(len(self.items)):
            self.model.add(sum(x[i, j] for j in range(len(self.capacities))) <= 1)

        # only add these constraints if toxic flag is set
        if self.activate_toxic:
            for i in range(len(self.items)):
                for j in range(len(self.capacities)):
                    if self.items[i].toxic: # toxic item selected means truck must also be marked as toxic 
                        self.model.add_implication(x[i, j], y[j])
                    else: # if truck has a toxic item non toxic items cannot be selected
                        self.model.add_implication(y[j], ~x[i, j])

        # maximize the sum of packed items values per truck
        self.model.maximize(sum(self.values[i] * x[i, j] for i in range(len(self.items)) for j in range(len(self.capacities))))

        status = self.solver.solve(self.model)

        assert status in [OPTIMAL, FEASIBLE]

        trucks = []
        for j in range(len(self.capacities)):  # for each truck
            trucks.append([])
            for i in range(len(self.items)):  # for each item
                if (self.solver.value(x[i, j]) == 1):  # if item is packed add it to the trucks knapsack
                    trucks[j].append(self.items[i])

        return Solution(trucks=trucks)  # empty solution
