from model import Coloring_Network
from evaluate import evaluate_boosted
from csp_utils import CSP_Instance, coloring_language

import data_utils
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='The model directory of a trained network')
    parser.add_argument('-t', '--t_max', type=int, default=40, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None, help='A path to a training set of graphs in the NetworkX adj_list format. If left unspecified, random instances are used.')
    parser.add_argument('-v', '--n_variables', type=int, default=400, help='Number of variables in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-c', '--n_clauses', type=int, default=1000, help='Number of clauses in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-i', '--n_instances', type=int, default=5000, help='Number of instances for training. Only used when --data_path is not specified.')
    args = parser.parse_args()

    if args.data_path is not None:
        print('loading graphs...')
        graphs = data_utils.load_graphs(os.path.join(args.data_path, '**'))
        instances = [CSP_Instance.graph_to_csp_instance(g, coloring_language, 'NEQ') for g in graphs]
    else:
        print(f'Generating {args.n_instances} training instances')
        instances = [CSP_Instance.generate_random(args.n_variables, args.n_clauses, coloring_language) for _ in tqdm(range(args.n_instances))]

    network = Coloring_Network.load(args.model_dir)
    evaluate_boosted(network, instances, args.t_max, args.attempts)


if __name__ == '__main__':
    main()
