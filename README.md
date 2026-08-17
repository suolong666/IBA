# IBA: Information-Gain Budget Allocation

IBA is a two-stage sequential recommendation framework based on semantic IDs.

## Setup

~~~bash
conda create -n iba python=3.9
conda activate iba
pip install -r requirements.txt
~~~

Install the PyTorch build that matches your CUDA version before installing the remaining packages.

## Data Preparation and RQ-VAE Index Generation

Prepare each dataset in data/<DATASET>/:

~~~text
data/<DATASET>/
  <DATASET>.item.json
  <DATASET>.inter.json
  <DATASET>.index.json
~~~

First, use an RQ-VAE pipeline to generate the semantic-ID index JSON file:

~~~text
data/<DATASET>/<DATASET>.index.json
~~~

The RQ-VAE implementation is not included in this repository. Copy its generated index file into the dataset directory before training.

## Stage 1 Training

From the project root, run:

~~~bash
bash IBA/train_stage1.sh
~~~

To modify experiment settings, edit the variable defaults in IBA/train_stage1.sh.

## Stage 2 Training

Stage 2 initializes from a Stage 1 checkpoint and trains the learned budget-allocation module:

~~~bash
bash IBA/train_stage2_learned_budget.sh
~~~

Set BASELINE_CKPT in IBA/train_stage2_learned_budget.sh to the Stage 1 checkpoint directory you want to use.

## Evaluation

Evaluate a fine-tuned checkpoint with:

~~~bash
bash IBA/test.sh
~~~

Set CKPT_PATH in IBA/test.sh to the checkpoint directory to evaluate.

## Citation

~~~bibtex
@article{yang2026where,
  title={Where Reasoning Matters: Rethinking Latent Reasoning in Semantic ID-based Generative Recommendation},
  author={Yang, Shangxin and Gao, Min and Wang, Zongwei and Yu, Junliang},
  year={2026}
}
~~~
