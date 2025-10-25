import math
from collections import defaultdict

import networkx as nx
from data_schema import Donation, Solution
from database import TransplantDatabase
from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL, CpModel, CpSolver


class CycleLimitingCrossoverTransplantSolver:
    def __init__(self, database: TransplantDatabase) -> None:
        """
        Constructs a new solver instance, using the instance data from the given database instance.
        :param Database database: The organ donor/recipients database.
        """

        self.database = database
        
        self.donors = self.database.get_all_donors()
        self.recipients = self.database.get_all_recipients()

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
        if timelimit <= 0.0:
            return Solution(donations=[])
        if timelimit < math.inf:
            self.solver.parameters.max_time_in_seconds = timelimit

        model = CpModel()

        # get all cycles with a length of at most 3
        cycles = [cycle for cycle in nx.simple_cycles(self.G, length_bound=3)]

        if not cycles:
            return Solution(donations=[])
        
        # precompute donors and recipients in a cycle for every cycle 
        cycle_donors = []
        cycle_recipients = []
        for cycle in cycles:
            donors_in_cycle = []
            recipients_in_cycle = list(cycle)

            n = len(cycle)
            for i in range(n):
                r_i = cycle[i]
                r_j = cycle[(i + 1) % n]

                if not self.G.has_edge(r_i, r_j):
                    break

                donors_in_cycle.append(self.G[r_i][r_j]['donor'])
            
            cycle_donors.append(donors_in_cycle)
            cycle_recipients.append(recipients_in_cycle)

        # decision variable that is true if a cycle is selected
        x = {
            i: model.new_bool_var(f"x_{i}") for i in range(len(cycles))
        }

        donor_to_cycle_indices = defaultdict(list)
        for i, donors_in_cycle in enumerate(cycle_donors):
            for donor in donors_in_cycle:
                donor_to_cycle_indices[donor].append(i)

        # 1. A donor can donate only once.
        for donor, indices in donor_to_cycle_indices.items():
            if indices:
                model.add_at_most_one(x[i] for i in indices)

        recipient_to_cycle_indices = defaultdict(list)
        for i, recipients_in_cycle in enumerate(cycle_recipients):
            for recipient in recipients_in_cycle:
                recipient_to_cycle_indices[recipient].append(i)
        
        # 2. A recipient can receive only one organ.
        for recipient, indices in recipient_to_cycle_indices.items():
            if indices:
                model.add_at_most_one(x[i] for i in indices)

        # maximize the length of the cycles
        model.maximize(sum(len(cycles[i]) * x[i] for i in range(len(cycles))))

        status = self.solver.solve(model)

        assert status in [OPTIMAL, FEASIBLE]

        donations = []
        for i in range(len(cycles)):
            if self.solver.value(x[i]) == 1:
                cycle = cycles[i]
                for j in range(len(cycle)):
                    r_i = cycle[j]
                    r_j = cycle[(j + 1) % len(cycle)]

                    if self.G.has_edge(r_i, r_j):
                        donor = self.G[r_i][r_j]['donor']
                        recipient = r_j
                        donations.append(Donation(donor=donor, recipient=recipient))

        return Solution(donations=donations)
