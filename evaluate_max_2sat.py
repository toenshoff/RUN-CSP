from model import Max_2SAT_Network
from evaluate import evaluate_boosted
from csp_utils import CSP_Instance, max_2sat_language

import data_utils
import argparse
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='The model directory of a trained network')
    parser.add_argument('-t', '--t_max', type=int, default=40, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None, help='A path to a training set of formulas in the DIMACS cnf format. If left unspecified, random instances are used.')
    parser.add_argument('-v', '--n_variables', type=int, default=100, help='Number of variables in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-c', '--n_clauses', type=int, default=300, help='Number of clauses in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-i', '--n_instances', type=int, default=5000, help='Number of instances for training. Only used when --data_path is not specified.')
    args = parser.parse_args()

    if args.data_path is not None:
        print('loading cnf formulas...')
        formulas = data_utils.load_formulas(os.path.join(args.data_path, '**'))
        instances = [CSP_Instance.cnf_to_instance(f) for f in formulas]
    else:
        print(f'Generating {args.n_instances} training instances')
        instances = [CSP_Instance.generate_random(args.n_variables, args.n_clauses, max_2sat_language) for _ in tqdm(range(args.n_instances))]

    # construct and train new network
    network = Max_2SAT_Network.load(args.model_dir)
    evaluate_boosted(network, instances, t_max=args.t_max)


if __name__ == '__main__':
    main()
