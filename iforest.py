# Reproducing Table 3 of "Isolation Forest" (Liu, Ting & Zhou, 2008)
#
# For each dataset I:
#   1. load the data (X = the data points, y = 1 for outlier, 0 for normal)
#   2. run Isolation Forest 10 times (each run is a bit random)
#   3. measure how well it finds the outliers with AUC (1.0 = perfect, 0.5 = guessing)
#   4. compare my average AUC with the number in the paper
#
# Run it with:  python iforest.py

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Look for the data and save the results in the same folder as this script,
# so it works no matter which folder the terminal is in.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Settings from the paper
N_TREES = 100       # number of trees
SAMPLE_SIZE = 256   # number of points each tree gets to see
RUNS = 10           # how many times I repeat the experiment

# Dataset name: (file name, AUC reported in the paper's Table 3)
DATASETS = {
    "Http":        ("16_http.npz",         1.00),
    "ForestCover": ("10_cover.npz",        0.88),
    "Smtp":        ("34_smtp.npz",         0.88),
    "Shuttle":     ("32_shuttle.npz",      1.00),
    "Mammography": ("23_mammography.npz",  0.86),
    "Annthyroid":  ("2_annthyroid.npz",    0.82),
    "Satellite":   ("30_satellite.npz",    0.71),
    "Pima":        ("29_Pima.npz",         0.67),
    "Breastw":     ("4_breastw.npz",       0.99),
    "Ionosphere":  ("18_Ionosphere.npz",   0.85),
}

results = []

for name, (file_name, paper_auc) in DATASETS.items():
    data = np.load("data/" + file_name)
    X = data["X"]
    y = data["y"]

    aucs = []
    for run in range(RUNS):
        model = IsolationForest(n_estimators=N_TREES, max_samples=SAMPLE_SIZE, random_state=run)
        model.fit(X)
        # scikit-learn gives outliers LOW scores, so I add a minus sign
        # to make outliers get HIGH scores, which is what AUC expects.
        scores = -model.score_samples(X)
        aucs.append(roc_auc_score(y, scores))

    results.append({
        "Dataset": name,
        "Points": len(X),
        "Paper AUC": paper_auc,
        "My AUC": round(np.mean(aucs), 3),
        "Spread (std)": round(np.std(aucs), 3),
    })
    print(name, "done")

table = pd.DataFrame(results)
print()
print(table.to_string(index=False))
table.to_csv("results.csv", index=False)
