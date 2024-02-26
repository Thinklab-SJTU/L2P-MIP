## L2P-MIP

This repo provides code for 'L2P-MIP: Learning to Presolve for Mixed Integer Programming', which is accepted by ICLR 2024.

## Installation

- Environment (tested on)
  - numpy==1.21.6
  - PySCIPOpt==3.5.0
  - requests==2.28.1
  - scipy==1.7.3
  - torch==1.10.0+cu113
  - tqdm==4.64.1
  - ecole==0.7.3
  - PyYAML==6.0
  - pandas==1.3.5

## Dataset preparation

1. Download the dataset from [corlat](https://coral.ise.lehigh.edu/data-sets/mixed-integer-instances/), [mik](https://atamturk.ieor.berkeley.edu/data/conic.sch/), [mirp](https://mirplib.scl.gatech.edu/home), [item_placement](https://www.ecole.ai/2021/ml4co-competition/), [load_balancing](https://www.ecole.ai/2021/ml4co-competition/), [anonymous](https://www.ecole.ai/2021/ml4co-competition/).

2. Unzip the dataset and create the `data/` folder to put it. (We provide a small case in the `example_case/` folder)

## Example Run

You can try out the following 5 steps one by one.

1. Data generator:

   `annealing_generator.py`

   Generator uses the simulated annealing algorithm to search the near optimal presolver parameters for each instance.
   The presolver parameter includes priority, time, and round (14 parameter for each, 42 in total).
   We use multi-processing to improve the efficiency. The annealing_generator.py is used in medium dataset and use the
   time gap as the reward.

   Input: the path of MIP instances, e.g. `data/medium/corlat/train/`

   Output: the log of searching, e.g. `results/annealing_results_corlat.txt` (instance --> presolve parameter)

2. Log processor:

   `read_log.py`

   The log processor is to better show the results of the generator,
   which turn the original ".txt" results to the ".csv" table for better illustration.
   Please note that this step is optional and can be skipped.

   Input: the log of searching, e.g. `results/annealing_results_corlat.txt`

   Output: the statistic table of searching, e.g. `results/annealing_results_corlat.csv`

3. Generate Ecole feature:

   `generate_ecole_feature.py`

   This python file will generate the Ecole feature for each instance.

   Example bash: python generate_ecole_feature.py corlat

   Input: the path of MIP instances, e.g. `data/medium/corlat/train/`; the path of csv file, e.g. `
   results/annealing_results_corlat.csv`

   Output: the path of ecole feature, e.g. `data/ecole_feature/item_placement/train/`

4. Data processor(require ecole features):

   `read_log_with_ecole_feature.py`

   Here we use additional ecole features of the MIP instances,
   and use the pair <ecole feature, presolve parameter> as the dataset for machine learning.

   Input: the log of searching, e.g. `results/annealing_results_corlat.txt`; the path of ecole feature, e.g. `
   data/ecole_feature/corlat/train/`

   Output: the dataset for machine learning, e.g. `generated_data/annealing_data_corlat_train/test.json` (<ecole
   feature, presolve parameter>)

5. Machine learning method:

   `run_seperated_p2rt.py`

   We use a neural network to learn the pairs <ecole feature, presolve parameter>.
   Please refer to the file for detailed description of arguments.

   Input: the dataset, e.g. `generated_data/annealing_data_corlat_train.json`

   Output: the log and model.