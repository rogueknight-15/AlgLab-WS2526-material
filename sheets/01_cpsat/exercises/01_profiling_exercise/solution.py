import itertools
import logging

# pip install networkx
import networkx as nx
from data_schema import ProblemInstance, Solution

# pip install ortools
from ortools.sat.python import cp_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)


def get_edge_weight(weighted_graph: ProblemInstance, u: str, v: str) -> int:
    """Retrieve the weight of the edge between two nodes."""
    for edge in weighted_graph.connections:
        if (edge.endpoint_a == u and edge.endpoint_b == v) or (
            edge.endpoint_a == v and edge.endpoint_b == u
        ):
            return edge.distance
    raise KeyError(f"Edge {u} - {v} not found in the graph")


def build_weighted_graph(instance: ProblemInstance) -> nx.Graph:
    """Build a NetworkX graph from the problem instance so we can use its shortest path implementation."""
    logging.info(
        "Building weighted graph with %d nodes and %d edges",
        len(instance.endpoints),
        len(instance.connections),
    )
    G = nx.Graph()
    
    # connections already store distance of edge
    # nodes are automatically added if not already present
    # if edge exists its only updated
    G.add_weighted_edges_from((edge.endpoint_a, edge.endpoint_b, edge.distance) for edge in instance.connections)

    return G


def distance(instance: ProblemInstance, u: str, v: str) -> int:
    """Calculate the shortest path distance between two endpoints in the network."""
    graph = build_weighted_graph(instance)
    return nx.shortest_path_length(graph, u, v, weight="weight")


class MaxPlacementsSolver:
    """
    A solver for the maximum number of placements problem using Google OR-Tools' CP-SAT solver.
    """

    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.model = cp_model.CpModel()

        # Create a boolean variable for each approved endpoint
        # It will be True if the (approved) endpoint is selected, False otherwise
        logging.info("Creating boolean variables for each endpoint")
        self.vars = {
            endpoint: self.model.new_bool_var(endpoint)
            for endpoint in instance.approved_endpoints
        }

        # compute graph only once
        graph = build_weighted_graph(instance)

        # precompute the shortest paths in a dictionary for fast lookup
        # start node is u
        # end node and length is a tuple (v, d) computed with nx.shortest_path_length()
        self.shortest_paths = {
            u: nx.shortest_path_length(graph, u, weight="weight") for u in instance.approved_endpoints
        }

        # Add constraints and objective to the model
        self._add_distance_constraints()
        self._set_objective()
        logging.info("Finished building the model")

    def _add_distance_constraints(self):
        """Add constraints to ensure selected endpoints are not too close."""
        logging.info("Adding distance constraints")
        for u, v in itertools.combinations(self.instance.approved_endpoints, 2):
            distance = self.shortest_paths[u][v]
            if distance < self.instance.min_distance_between_placements:
                self.model.Add(self.vars[u] + self.vars[v] <= 1)

    def _set_objective(self):
        """Set the objective to maximize the number of selected endpoints."""
        logging.info("Setting objective to maximize the number of selected endpoints")
        self.model.Maximize(sum(self.vars.values()))

    def solve(self, time_limit: float = 10) -> Solution:
        """Solve the optimization problem within the given time limit."""
        logging.info("Solving the model with a time limit of %d seconds", time_limit)
        # Create a solver instance
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        # Enable logging to stdout so we can see the progress
        solver.parameters.log_search_progress = True
        solver.parameters.log_to_stdout = True

        # Solve the model
        status = solver.solve(self.model)

        # Return the solution if one was found (either optimal or at least feasible)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logging.info("Successfully found a solution")
            # Retrieve the selected endpoints by querying the value of the variables
            selected_placement = [
                endpoint for endpoint in self.vars if solver.Value(self.vars[endpoint])
            ]
            return Solution(selected_placements=selected_placement)
        logging.warning("No solution found within the time limit.")
        raise RuntimeError("No solution found within the time limit.")


if __name__ == "__main__":
    # load instance
    instance = ProblemInstance.parse_file("instances/instance_50.json")
    # solve instance
    solver = MaxPlacementsSolver(instance)
    solution = solver.solve()
    print(solution)
