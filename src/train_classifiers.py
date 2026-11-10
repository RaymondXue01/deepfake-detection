"""Train simple classifiers (Logistic, SVM, RandomForest) on embeddings and report ROC-AUC.

Usage:
    python src/train_classifiers.py --embeddings data/embeddings_vit_small.npz --out outputs/models
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import joblib


def encode_labels(labels):
    # map unique strings to 0/1
    u = sorted(list(set(labels.tolist())))
    mapping = {v: i for i, v in enumerate(u)}
    y = np.array([mapping[x] for x in labels.tolist()])
    return y, mapping


def fit_and_eval(X_train, X_test, y_train, y_test, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    models = {
        'logistic': LogisticRegression(max_iter=2000),
        'svm': SVC(probability=True),
        'rf': RandomForestClassifier(n_estimators=200),
    }
    results = {}
    for name, m in models.items():
        print('Training', name)
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        acc = accuracy_score(y_test, m.predict(X_test))
        results[name] = {'auc': float(auc), 'acc': float(acc)}
        joblib.dump(m, outdir / f'{name}.joblib')
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(outdir / 'roc_curves.png', dpi=150)
    plt.close()

    with open(outdir / 'results.txt', 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: auc={v['auc']:.4f}, acc={v['acc']:.4f}\n")
    print('Saved models and results to', outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--out', default='outputs/models')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()
    out = Path(args.out)
    data = np.load(args.embeddings, allow_pickle=True)
    X = data['embeddings']
    labels = data['labels']
    y, mapping = encode_labels(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    fit_and_eval(X_train, X_test, y_train, y_test, out)


if __name__ == '__main__':
    main()
