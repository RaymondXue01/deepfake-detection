"""PCA analysis on DINO embeddings with visualization and optional classification.

Usage:
    python src/pca_analysis.py --embeddings data/embeddings_vit_small.npz --outdir outputs/pca --n-components 50 --classify --k 10
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def encode_labels(labels):
    u = sorted(list(set(labels.tolist())))
    mapping = {v: i for i, v in enumerate(u)}
    y = np.array([mapping[x] for x in labels.tolist()])
    return y, mapping


def plot_explained_variance(pca, outpath: Path):
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(ev) + 1), ev, marker='o', label='Individual')
    plt.plot(np.arange(1, len(ev) + 1), cum, marker='x', label='Cumulative')
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.legend()
    plt.title('PCA Explained Variance')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_2d_proj(X2, labels, outpath: Path, title='PCA 2D'):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=labels, palette='tab10', s=8, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def train_logistic_on_pcs(X, y, k, outdir: Path, test_size=0.2):
    Xk = X[:, :k]
    X_train, X_test, y_train, y_test = train_test_split(Xk, y, test_size=test_size, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    acc = accuracy_score(y_test, clf.predict(X_test))
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Logistic (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title(f'ROC on top-{k} PCs')
    plt.savefig(outdir / f'roc_top{k}.png', dpi=150)
    plt.close()
    with open(outdir / f'metrics_top{k}.txt', 'w') as f:
        f.write(f'auc={auc:.6f}\naccuracy={acc:.6f}\n')
    return clf, auc, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--outdir', default='outputs/pca')
    parser.add_argument('--n-components', type=int, default=50)
    parser.add_argument('--classify', action='store_true', help='train logistic on top-k PCs')
    parser.add_argument('--k', type=int, default=10, help='number of PCs for classifier')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.embeddings, allow_pickle=True)
    X = data['embeddings']
    labels_raw = data['labels']
    y, mapping = encode_labels(labels_raw)

    n_comp = min(args.n_components, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    Xp = pca.fit_transform(X)
    np.save(out / 'pca_proj.npy', Xp)
    np.save(out / 'pca_components.npy', pca.components_)

    plot_explained_variance(pca, out / 'explained_variance.png')

    # 2D scatter using first 2 PCs
    plot_2d_proj(Xp[:, :2], labels_raw, out / 'pca_2d_by_label.png', title='PCA 2D by label')
    plot_2d_proj(Xp[:, :2], y, out / 'pca_2d_by_numeric.png', title='PCA 2D by numeric label')

    # Save mapping
    with open(out / 'label_mapping.txt', 'w') as f:
        for k, v in mapping.items():
            f.write(f'{k} -> {v}\n')

    if args.classify:
        clf, auc, acc = train_logistic_on_pcs(Xp, y, args.k, out, test_size=args.test_size)
        print(f'Logistic on top-{args.k} PCs: AUC={auc:.4f}, ACC={acc:.4f}')

    print('PCA analysis completed, outputs in', out)


if __name__ == '__main__':
    main()
