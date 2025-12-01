# COCO Implementation

## Requirements

- Python 3.7
- PyTorch 1.12
- psycopg2
- Ray 2.4.0

## Datasets

We build the training databases following the instructions from the repositories [zero-shot-cost-estimation](https://github.com/DataManagementLab/zero-shot-cost-estimation) and [PRICE](https://github.com/StCarmen/PRICE). The resulting collection includes the following schemas:

```text
['basketball', 'walmart', 'financial', 'movielens',
 'carcinogenesis', 'accidents', 'tournament',
 'employee', 'geneea', 'genome', 'seznam', 'fhnk', 'consumer',
 'ssb', 'hepatitis', 'credit', 'baseball', 'tpch', 'stats', 'imdb',
 'chembl', 'ergastf1', 'grants', 'legalacts', 'sap', 'talkingdata','tpcds']
```

## Feature Preparation

Extracting features for COCO can take some time. Run the following scripts in order:

1. `python ./src/tool/meta_info.py` – collect basic metadata.
2. `python ./src/tool/column_data_distribution.py` – compute column-wise data-distribution statistics.
3. `python ./src/tool/merge.py` – merge the features obtained in the previous steps.
4. `python ./src/tool/get_column_stats.py` – sample data used by SQLGen for query generation.

## Running COCO

First, modify the configuration of PostgreSQL in GenConfig/gen_config.py and then execute the test workload to obtain the test set:

```bash
python ./src/runSQL.py
```
Generating some random queries for warm-up:

```bash
python ./src/generate.py --target-per-db 10 --path ./TestWorkload/warmup_workload.pkl
```
Generating random source workloads for validation. It will take a considerable amount of time to complete (optionally):

```bash
python ./src/generate.py --target-per-db 100 --path ./TestWorkload/random_source_workload.pkl
```

Then start COCO training and roll-out evaluation:

```bash
python main.py
```
