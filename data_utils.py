import numpy as np
import networkx as nx
import os
import glob
from tqdm import tqdm


def load_graphs(path):
    """
    Loads the graphs from all '.adj' files in NetworkX adjacency list format
    :param path: The pattern under which to look for .adj files
    :return: A list of NetworkX graphs
    """
    paths = glob.glob(os.path.join(path, '*.adj'), recursive=True)
    graphs = [nx.read_adjlist(p) for p in tqdm(paths)]
    return graphs


def write_graphs(graphs, path):
    if not os.path.exists(path):
        os.mkdir(path)

    print(f'saving graphs at {path}')
    existing_files = glob.glob(os.path.join(path, '*.adj'))
    num_exist = len(existing_files)

    for i, g in enumerate(graphs):
        graph_path = os.path.join(path, f'{num_exist+i}.adj')
        nx.write_adjlist(g, graph_path)


def load_dimacs_cnf(path):
    """
    Loads a cnf formula from a file in dimacs cnf format
    :param path: the path to a .cnf file in dimacs format
    :return: The formula as a list of lists of signed integers. 
             I.E. ((X1 or X2) and (not X2 or X3)) is [[1, 2], [-2, 3]]
    """
    file = open(path, 'r')
    f = []
    for line in file:
        s = line.split()
        if not s[0] == 'c' and not s[0] == 'p':
            assert(s[-1] == '0')
            clause = [int(l) for l in s[:-1]]
            f.append(clause)
    file.close()
    return f


def write_dimacs_cnf(f, path):
    """
    Stores a cnf formula in the dimacs cnf format
    :param f: The formula as a list of lists of signed integers.
    :param path: The path to a file in which f is will be stored
    """
    file = open(path, 'w')
    
    num_v = np.max([np.max(np.abs(clause)) for clause in f])
    num_c = len(f)
    file.write(f'p cnf {num_v} {num_c}\n')

    for clause in f:
        line = ''
        for l in clause:
            line += f'{l} '
        line += '0\n'
        file.write(line)
    file.close()
    return f


def load_formulas(path):
    """ Loads cnf formulas from all .cnf files found under the pattern 'path' """
    paths = glob.glob(os.path.join(path, '**/*.cnf'), recursive=True)
    formulas = [load_dimacs_cnf(p) for p in tqdm(paths)]
    return formulas


def write_formulas(formulas, path):
    """ Writes a list of cnf formulas to the location in 'path' """
    for i, f in enumerate(formulas):
        write_dimacs_cnf(f, os.path.join(path, f'{i}.cnf'))
