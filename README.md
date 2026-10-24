# Reproducing "Isolation Forest" (Liu, Ting & Zhou, 2008)

This project reproduces the main experiment (Table 3) of the Isolation Forest paper,
using scikit-learn's `IsolationForest`.

## How to run

```bash
pip install -r requirements.txt
python iforest.py
```

It takes about one minute and saves the results to `results.csv`.

## Files

- `iforest.py`: the whole experiment
- `data/`: the datasets, from the ADBench GitHub repository (https://github.com/Minqi824/ADBench), which collects the ODDS benchmark datasets
- `results.csv`: the output

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 100 | Number of trees (paper default) |
| `max_samples` | 256 | Points each tree gets to see (paper default) |
| `random_state` | 0 to 9 | 10 runs with different randomness, averaged |
| Evaluation | fit and score on the full dataset | Same as the paper |
| Metric | AUC | Same as the paper |

## Results

| Dataset | Points | Paper AUC | My AUC | Spread (std) |
|---|---|---|---|---|
| Http | 567,498 | 1.00 | 1.000 | 0.000 |
| ForestCover | 286,048 | 0.88 | 0.877 | 0.035 |
| Smtp | 95,156 | 0.88 | 0.904 | 0.006 |
| Shuttle | 49,097 | 1.00 | 0.997 | 0.000 |
| Mammography | 11,183 | 0.86 | 0.861 | 0.007 |
| Annthyroid | 7,200 | 0.82 | 0.818 | 0.016 |
| Satellite | 6,435 | 0.71 | 0.701 | 0.019 |
| Pima | 768 | 0.67 | 0.671 | 0.007 |
| Breastw | 683 | 0.99 | 0.987 | 0.001 |
| Ionosphere | 351 | 0.85 | 0.846 | 0.006 |

Two of the paper's 12 datasets are not included: Mulcross (artificial data the
paper generated itself) and Arrhythmia (not in the ADBench collection).
