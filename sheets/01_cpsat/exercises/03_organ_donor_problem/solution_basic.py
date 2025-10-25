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

        self.donors = self.database.get_all_donors()
        self.recipients = self.database.get_all_recipients()
        self.pairs = [ 
            (d, self.database.get_partner_recipient(d)) 
            for d in self.donors
        ]

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
        n = len(self.pairs)

        donor_to_compatible_recipient = {
            donor.id: set(self.database.get_compatible_recipients(donor))
            for donor, _ in self.pairs
        }

        x = {}
        for i, (d_i, _) in enumerate(self.pairs):
            compatible_recipients = donor_to_compatible_recipient[d_i.id]
            for j, (_, r_j) in enumerate(self.pairs):
                if i != j and r_j in compatible_recipients :
                    x[i, j] = model.new_bool_var(f"x_{i}_{j}")

        unique_recipients = list({recipient for _, recipient in self.pairs})
        recipients_to_pairs = {
            recipient: [i for i, (_, r_i) in enumerate(self.pairs) if r_i.id == recipient.id] for recipient in unique_recipients
        }

        donor_outgoing = {
            i: [x[i, j] for j in range(n) if (i, j) in x] for i in range(n)
        }
        recipient_incoming = {
            j: [x[i, j] for i in range(n) if (i, j) in x] for j in range(n)
        }

        for i in range(n):
            if donor_outgoing[i]:
                model.add_at_most_one(donor_outgoing[i])

            if donor_outgoing[i] or recipient_incoming[i]:
                model.add(sum(donor_outgoing[i]) <= sum(recipient_incoming[i]))

        for r in unique_recipients:
            bool_vars_recipient = [
                x[i, j]
                for i in range(n)
                for j in recipients_to_pairs[r]
                if (i, j) in x
            ]

            if bool_vars_recipient:
                model.add_at_most_one(bool_vars_recipient)

        model.maximize(sum(x[i, j] for (i, j) in x))

        status = self.solver.solve(model)

        assert status in [OPTIMAL, FEASIBLE]

        donations = []
        for (i, j) in x:
            if self.solver.value(x[i, j]) == 1:
                donor = self.pairs[i][0]
                recipient = self.pairs[j][1]
                donations.append(Donation(donor=donor, recipient=recipient))

        return Solution(donations=donations)
