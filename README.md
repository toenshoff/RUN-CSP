# RUN-CSP
This repository contains a Tensorflow implementation of RUN-CSP,
a recurrent neural network architecture for solving Constraint Satisfaction Problems (https://arxiv.org/abs/1909.08387).

We provide a tool to automatically train a RUN-CSP instance for any fixed constraint language.
A Constraint Language is represented as a JSON file that specifies a domain size and the relations.
An example is provided in example_language.json. To train a model for this language use:

```python3 train.py -l example_language.json -m models/example_model```

This script will generate random instances for the specified Constraint Language and use them to train RUN-CSP.\
The trained model is stored in the given model directory.
To evaluate the result on random instances, use the evaluation script:

```python3 evaluate.py -m models/example_model```

This script also boosts the performance by performing multiple parallel runs on each instance (64 by default) an using the best result

Additionally, we provide training and evaluation scripts for specific versions of RUN-CSP for the Max-2SAT, Max-3-Col and Max-IS problems.
These scripts do not require a language specification and also train on random data by default.
However, they also allow training and evaluation data to be loaded from disk.
The scripts for Max-3-Col and Max-IS can be trained on graphs stored in the adjacency list format from NetworkX.
For example, use

```python3 train_coloring.py -m models/coloring_model -d data/3Col_100 ```

to train a RUN-CSP model for maximum 3-coloring on all '.adj' files in the 'data/3Col_100' directory.
The repository contains 3 graph datasets for 3 colorability with different graph sizes: 3Col_100, 3Col_200, 3Col_400.
These each contain 2500 pairs of graphs, where one graph is 3-colorable and the other graph has one additional edge and is non-3-colorable.

The scripts for Max-2SAT can run with 2-cnf formulas specified in the DIMACS cnf format.