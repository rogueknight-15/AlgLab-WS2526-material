import math

import networkx as nx
from data_schema import Donation, Solution
from database import TransplantDatabase
from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL, CpModel, CpSolver

class CrossoverTransplantSolver:
    def __init__(self, database: TransplantDatabase) -> None:
        """
        Constructs a new solver instance, using the instance data from the given database instance.
        :param Database database: The organ donor/recipients database.
        """
        self.database = database

        # list of all donors
        self.donors = self.database.get_all_donors()
        # list of all recipients
        self.recipients = self.database.get_all_recipients()

        # excact implementation of graph abstraction in the note section of the task
        self.G = nx.DiGraph()

        for recipient in self.recipients:
            self.G.add_node(recipient)

        for r_i in self.recipients:
            partner_donors = self.database.get_partner_donors(r_i)
            for d_k in partner_donors:
                compatible_recipients = self.database.get_compatible_recipients(d_k)
                for r_j in compatible_recipients:
                    if r_i != r_j:
                        self.G.add_edge(r_i, r_j, donor=d_k)

        self.solver = CpSolver()
        self.solver.parameters.log_search_progress = True
    
    def optimize(self, timelimit: float = math.inf) -> Solution:
        """
        Solves the constraint programming model and returns the optimal solution (if found within time limit).
        :param timelimit: The maximum time limit for the solver.
        :return: A list of Donation objects representing the best solution, or None if no solution was found.
        """
        if timelimit <= 0.0:
            return Solution(donations=[])
        if timelimit < math.inf:
            self.solver.parameters.max_time_in_seconds = timelimit

    
        model = CpModel()

        # decision variables x representing which donor i donates to recipient j
        x = {}
        for i, j in self.G.edges:
            x[i, j] = model.new_bool_var(f"x_{i}_{j}")

        # 1. Donor can donate only once
        for i in self.G.nodes:
            out_vars = [x[i, j] for j in self.G.successors(i)]
            model.add_at_most_one(out_vars)
        
        # 2. Recipient can receive only one donation/organ
        for j in self.G.nodes:
            in_vars = [x[i, j] for i in self.G.predecessors(j)]
            model.add_at_most_one(in_vars)

        # 3. Donor only donates if their own recipient receives am organ
        for i in self.G.nodes:
            out_vars = [x[i, j] for j in self.G.successors(i)]
            in_vars = [x[k, i] for k in self.G.predecessors(i)]
            model.add(sum(out_vars) == sum(in_vars))

        # maximize number of transplantations
        model.maximize(sum(x[i, j] for (i, j) in self.G.edges))

        status = self.solver.solve(model)

        assert status in [OPTIMAL, FEASIBLE]

        # get the donations for the solution
        donations = []
        for (i, j) in x:
            if self.solver.value(x[i, j]) == 1:
                donor = self.G[i][j]['donor']
                recipient = j
                donations.append(Donation(donor=donor, recipient=recipient))

        return Solution(donations=donations)
