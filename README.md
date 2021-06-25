# Credit Card Fraud Detection

See: [notebook.ipynb](notebook.ipynb)

## Contents

1. Download datasets
2. Setup Project
3. Start JupyterLab

---

## Download datasets

1. Kaggle Datasets: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Add `creditcard.csv` to `.gitignore`

## Setup Project

- Conda Documentation: [Reinstall packages from an export file](https://docs.conda.io/projects/conda/en/latest/commands/list.html#Output,%20Prompt,%20and%20Flow%20Control%20Options)
- Conda Env: [Create a conda environment](https://github.com/rurumimic/python#create-a-conda-environment)

### Install all packages

```bash
conda create -n credit --file package-list.txt
```

### Manually

Create a env:

```bash
pyenv activate miniconda3-latest
conda create -n credit python=3.9.2
```

Activate and deactivate:

```bash
pyenv activate miniconda3-latest/envs/credit
pyenv deactivate
```

## Start JupyterLab

Install package:

```bash
conda install -c conda-forge jupyterlab
```

Start Lab:

```bash
jupyter lab
```
