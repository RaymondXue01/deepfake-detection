"""Full analysis: normalization, PCA, UMAP, k-means, and supervised baselines.

Usage:
    python src/full_analysis.py --embeddings data/embeddings_vit_small.npz --outdir outputs/full --n-pca 50 --k-pc 10
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


def cluster_purity(labels_true, labels_pred):
    # purity = sum_{clusters} max_j |C_i intersect L_j| / N
    from collections import Counter, defaultdict
    cluster_to_counts = defaultdict(Counter)
    for t, p in zip(labels_true, labels_pred):
        cluster_to_counts[p][t] += 1
    total = 0
    for p, counter in cluster_to_counts.items():
        total += max(counter.values())
    return total / len(labels_true)


def plot_explained_variance(pca, outpath: Path):
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(ev) + 1), ev, marker='o', label='Individual')
    plt.plot(np.arange(1, len(ev) + 1), cum, marker='x', label='Cumulative')
    plt.xlabel('PC')
    plt.ylabel('Explained variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_2d(X2, labels, outpath: Path, title='2D projection'):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=labels, palette='tab10', s=8, alpha=0.8, legend='brief')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def eval_classifiers(X, y, outdir: Path, prefix=''):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        'logistic': LogisticRegression(max_iter=2000),
        'svm': SVC(probability=True),
        'rf': RandomForestClassifier(n_estimators=200),
        'knn': KNeighborsClassifier(n_neighbors=5),
    }
    records = []
    plt.figure()
    for name, m in models.items():
        m.fit(X_train, y_train)
        if hasattr(m, 'predict_proba'):
            prob = m.predict_proba(X_test)[:, 1]
        else:
            # fallback to decision_function
            try:
                prob = m.decision_function(X_test)
            except Exception:
                prob = m.predict(X_test)
        auc = roc_auc_score(y_test, prob)
        pr = average_precision_score(y_test, prob)
        acc = accuracy_score(y_test, m.predict(X_test))
        records.append({'model': name, 'auc': float(auc), 'pr_auc': float(pr), 'acc': float(acc), 'prefix': prefix})
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curves {prefix}')
    plt.legend()
    plt.savefig(outdir / f'roc_{prefix}.png')
    plt.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--outdir', default='outputs/full')
    parser.add_argument('--n-pca', type=int, default=50)
    parser.add_argument('--k-pc', type=int, default=10)
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.embeddings, allow_pickle=True)
    X = data['embeddings']
    labels = data['labels']
    # map labels to 0/1
    u = sorted(list(set(labels.tolist())))
    mapping = {v: i for i, v in enumerate(u)}
    y = np.array([mapping[x] for x in labels.tolist()])

    # L2-normalize embeddings for distance-based methods
    Xn = normalize(X, norm='l2')
    np.save(out / 'embeddings_normalized.npy', Xn)

    # PCA
    n_pca = min(args.n_pca, Xn.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    Xp = pca.fit_transform(Xn)
    np.save(out / 'pca_proj.npy', Xp)
    np.save(out / 'pca_components.npy', pca.components_)
    plot_explained_variance(pca, out / 'explained_variance.png')

    # 2D PCA plot
    plot_2d(Xp[:, :2], labels, out / 'pca_2d_by_label.png', title='PCA 2D by label')

    # t-SNE on PCA-reduced features
    print('Running t-SNE...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    Xt = tsne.fit_transform(Xp)
    np.save(out / 'tsne_2d.npy', Xt)
    plot_2d(Xt, labels, out / 'tsne_2d_by_label.png', title='t-SNE 2D by label')

    # UMAP on PCA-reduced features
    print('Running UMAP...')
    reducer = umap.UMAP(n_components=2, random_state=42)
    Xu = reducer.fit_transform(Xp)
    np.save(out / 'umap_2d.npy', Xu)
    plot_2d(Xu, labels, out / 'umap_2d_by_label.png', title='UMAP 2D by label')

    # k-means for k=2..5
    k_records = []
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42).fit(Xp)
        sil = silhouette_score(Xp, km.labels_)
        purity = cluster_purity(labels.tolist(), km.labels_.tolist())
        k_records.append({'k': k, 'silhouette': float(sil), 'purity': float(purity)})
        # plot clusters on PCA/UMAP
        plot_2d(Xp[:, :2], km.labels_, out / f'pca_2d_k{k}.png', title=f'PCA 2D k={k}')
        plot_2d(Xt, km.labels_, out / f'tsne_2d_k{k}.png', title=f't-SNE 2D k={k}')
        plot_2d(Xu, km.labels_, out / f'umap_2d_k{k}.png', title=f'UMAP 2D k={k}')

    pd.DataFrame(k_records).to_csv(out / 'kmeans_metrics.csv', index=False)

    # Supervised baselines
    results = []
    # representation: full normalized embeddings
    results += eval_classifiers(Xn, y, out, prefix='emb_full')
    # representation: top-k PCs
    kpc = min(args.k_pc, Xp.shape[1])
    results += eval_classifiers(Xp[:, :kpc], y, out, prefix=f'pc_top{kpc}')

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(out / 'supervised_results.csv', index=False)
    with open(out / 'label_mapping.txt', 'w') as f:
        for k, v in mapping.items():
            f.write(f'{k} -> {v}\n')

    print('Full analysis completed. Outputs saved to', out)


if __name__ == '__main__':
    main()
