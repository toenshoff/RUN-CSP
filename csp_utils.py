import numpy as np
import networkx as nx
import json
import itertools

from tqdm import tqdm


class Constraint_Language:
    """ Class to represent a fixed Constraint Language """

    def __init__(self, domain_size, relations):
        """
        :param domain_size: Size of the underlying domain
        :param relations: A dict specifying the relations of the language. This also specifies a name for each relation.
                          I.E {'XOR': [[0, 1], [1, 0]], 'AND': [[1,1]]}
        """
        self.domain_size = domain_size
        self.relations = relations
        self.relation_names = list(relations.keys())

        # compute characteristic matrices for each relation
        self.relation_matrices = dict()
        for n, r in self.relations.items():
            M = np.zeros((self.domain_size, self.domain_size), dtype=np.float32)
            idx = np.array(r)
            M[idx[:, 0], idx[:, 1]] = 1.0
            self.relation_matrices[n] = M

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'domain_size': self.domain_size, 'relations': self.relations}, f, indent=4)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        language = Constraint_Language(data['domain_size'], data['relations'])
        return language


# define constant constraint languages for Vertex Coloring, Independent Set and Max2Sat
coloring_language = Constraint_Language(domain_size=3,
                                        relations={'NEQ': [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]})

is_language = Constraint_Language(domain_size=2,
                                  relations={'NAND': [[0, 0], [0, 1], [1, 0]]})

max_2sat_language = Constraint_Language(domain_size=2,
                                        relations={'OR': [[0, 1], [1, 0], [1, 1]],
                                                   'IMPL': [[0, 0], [0, 1], [1, 1]],
                                                   'NAND': [[0, 0], [0, 1], [1, 0]]})


class CSP_Instance:
    """ A class to represent a CSP instance """

    def __init__(self, language, n_variables, clauses):
        """
        :param language: A Constraint_Language object
        :param n_variables: The number of variables
        :param clauses: A dict specifying the clauses for each relation in the language.
                        I.E {'XOR': [[1,2], [5,4], [3,1]], 'AND': [[1,4], [2,5]]}
        """
        self.language = language
        self.n_variables = n_variables
        # assure clauses are un numpy format
        self.clauses = {r: np.int32(c) for r, c in clauses.items()}

        # compute number of clauses and degree of each variable
        all_clauses = list(itertools.chain.from_iterable(clauses.values()))
        variables, counts = np.unique(all_clauses, return_counts=True)
        degrees = np.zeros(shape=(n_variables), dtype=np.int32)
        for u, c in zip(variables, counts):
            degrees[u] = c

        self.degrees = degrees
        self.n_clauses = len(all_clauses)

    def count_conflicts(self, assignment):
        """
        :param assignment: A hard variable assignment represented as a list of ints of length n_variables.
        :return: The number of unsatisfied clauses in this instances
        """
        conflicts = 0
        matrices = self.language.relation_matrices
        for r, M in matrices.items():
            has_conflict = [M[assignment[u], assignment[v]] for [u, v] in self.clauses[r]]
            conflicts += len(has_conflict) - np.count_nonzero(has_conflict)
        return conflicts

    @staticmethod
    def merge(instances):
        """
        A static function that merges multiple CSP instances into one
        :param instances: A list of CSP instances
        :return: CSP instances that contains all given instances with shifted variables
        """
        language = instances[0].language

        clauses = {r: [] for r in language.relation_names}
        n_variables = 0

        for instance in instances:
            for r in language.relation_names:
                shifted = instance.clauses[r] + n_variables
                clauses[r].append(shifted)
            n_variables += instance.n_variables

        clauses = {r: np.vstack(c) for r, c in clauses.items()}
        merged_instance = CSP_Instance(language, n_variables, clauses)
        return merged_instance

    @staticmethod
    def batch_instances(instances, batch_size):
        """
        Static method to merge given instances into batches
        :param instances: A list of CSP instances
        :param batch_size: The batch size
        :return: A list of CSP instances that each consist of 'batch_size' many merged instances
        """
        n_instances = len(instances)
        n_batches = int(np.ceil(n_instances / batch_size))
        batches = []

        print('Combining instances in batches...')
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min(start + batch_size, n_instances)
            batch_instance = CSP_Instance.merge(instances[start:end])
            batches.append(batch_instance)

        return batches

    @staticmethod
    def generate_random(n_variables, n_clauses, language):
        """
        :param n_variables: Number of variables
        :param n_clauses: Number of clauses
        :param language: A Constraint Language
        :return: A random CSP Instance with the specified parameters. Clauses are sampled uniformly.
        """
        variables = list(range(n_variables))
        clauses = {r: [] for r in language.relation_names}
        relations = np.random.choice(language.relation_names, n_clauses)

        for i in range(n_clauses):
            clause = list(np.random.choice(variables, 2, replace=False))
            r = relations[i]
            clauses[r].append(clause)

        instance = CSP_Instance(language, n_variables, clauses)
        return instance

    @staticmethod
    def graph_to_csp_instance(graph, language, relation_name):
        """
        :param graph: A NetworkX graphs
        :param language: A Constraint Language
        :param relation_name: The relation name to assign to each edge
        :return: A CSP Instance representing the graph
        """
        adj = nx.linalg.adjacency_matrix(graph)
        n_variables = adj.shape[0]
        clauses = {relation_name: np.int32(np.transpose(adj.nonzero()))}

        instance = CSP_Instance(language, n_variables, clauses)
        return instance

    @staticmethod
    def cnf_to_instance(formula):
        """
        :param formula: A 2-cnf formula represented as a list of lists of ints.
                        I.e. ((X1 or X2) and (not X2 or X3)) is [[1, 2], [-2, 3]]
        :return: A CSP instance that represents the formula
        """
        def clause_type(clause):
            # returns the relation type for a given clause
            if clause[0] * clause[1] < 0:
                return 'IMPL'
            elif clause[0] > 0:
                return 'OR'
            else:
                return 'NAND'

        def normalize_2SAT_clauses(formula):
            # Transforms clauses of form [v, -u] to [-u, v]. This unifies the direction of all implication clauses.
            normalize_impl_clause = lambda c: [c[1], c[0]] if clause_type(c) == 'IMPL' and c[0] > 0 else c
            normed_formula = list(map(normalize_impl_clause, formula))
            return normed_formula

        formula = normalize_2SAT_clauses(formula)

        clauses = {t: [] for t in {'OR', 'IMPL', 'NAND'}}
        for c in formula:
            u = abs(c[0]) - 1
            v = abs(c[1]) - 1
            t = clause_type(c)
            clauses[t].append([u, v])

        n_variables = np.max([np.max(np.abs(clause)) for clause in formula])

        instance = CSP_Instance(max_2sat_language, n_variables, clauses)
        return instance
