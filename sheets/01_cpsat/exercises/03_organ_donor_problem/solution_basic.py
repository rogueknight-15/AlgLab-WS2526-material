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

        # list of all donor-recipient-pairs
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

        """
        Mathematical Model:
        maximize sum(x_(i,j)) for all i and j
        s.t.
            1. sum(x_(i, j)) <= 1 for all i
            2. sum(x_(i, j)) <= 1 for all j
            3. sum(x_(i, j)) == sum(x_(k, i)) for all i
            4. only one donor is willing to give its organ if there are multiple
        x_(i, j) ∈ {0, 1}
        i, j, k ∈ N

        3. means that if a donor donates their associated recipient receives an organ too
        4. is implemented with only considering unique recipients in the final solution
        """

        model = CpModel()
        n = len(self.pairs)

        # precompute compatible recipients for evry donor by id
        donor_to_compatible_recipient = {
            donor.id: set(self.database.get_compatible_recipients(donor))
            for donor, _ in self.pairs
        }

        # decision variables x representing which donor i donates to recipient j
        x = {}
        for i, (d_i, _) in enumerate(self.pairs):
            compatible_recipients = donor_to_compatible_recipient[d_i.id]
            for j, (_, r_j) in enumerate(self.pairs):
                if i != j and r_j in compatible_recipients :
                    x[i, j] = model.new_bool_var(f"x_{i}_{j}")

        # because recipients can be duplicates we only need the unique recipients
        unique_recipients = list({recipient for _, recipient in self.pairs})
        # map recipients to its donor-recipients-pairs
        recipients_to_pairs = {
            recipient: [i for i, (_, r_i) in enumerate(self.pairs) if r_i.id == recipient.id] for recipient in unique_recipients
        }

        # precompute outgoing donations for all donors i
        donor_outgoing = {
            i: [x[i, j] for j in range(n) if (i, j) in x] for i in range(n)
        }
        # precompute incoming donations for all recipients j
        recipient_incoming = {
            j: [x[i, j] for i in range(n) if (i, j) in x] for j in range(n)
        }

        for i in range(n):
            if donor_outgoing[i]:
                # 1. A donor can donate only once.
                model.add_at_most_one(donor_outgoing[i])

            if donor_outgoing[i] or recipient_incoming[i]:
                # 3. A donor is willing to donate only if their associated recipient receives an organ in exchange.
                model.add(sum(donor_outgoing[i]) == sum(recipient_incoming[i]))

        # 4. If a recipient has multiple willing donors, only one of them is willing to donate in the final solution.
        for r in unique_recipients:
            bool_vars_recipient = [
                x[i, j]
                for i in range(n)
                for j in recipients_to_pairs[r]
                if (i, j) in x
            ]

            if bool_vars_recipient:
                # 2. A recipient can receive only one organ.
                model.add_at_most_one(bool_vars_recipient) 

        # maximize number of transplantations
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
