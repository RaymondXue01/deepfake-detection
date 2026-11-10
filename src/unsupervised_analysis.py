"""Run PCA, UMAP, k-means on saved embeddings and produce plots/metrics."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True, help='npz file with embeddings, filenames, labels')
    parser.add_argument('--outdir', default='outputs/unsup')
    parser.add_argument('--n-components', type=int, default=50)
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.embeddings, allow_pickle=True)
    X = data['embeddings']
    labels = data['labels']

    print('Running PCA')
    pca = PCA(n_components=min(args.n_components, X.shape[1]))
    Xp = pca.fit_transform(X)
    np.save(out / 'pca_components.npy', pca.components_)

    print('Running t-SNE')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    Xt = tsne.fit_transform(Xp)
    np.save(out / 'tsne_2d.npy', Xt)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Xt[:, 0], y=Xt[:, 1], hue=labels, s=5, palette='tab10')
    plt.title('t-SNE 2D of embeddings')
    plt.legend(loc='best')
    plt.savefig(out / 'tsne_2d.png', dpi=150)
    plt.close()

    print('Running UMAP')
    reducer = umap.UMAP(n_components=2, random_state=42)
    Xu = reducer.fit_transform(Xp)
    np.save(out / 'umap_2d.npy', Xu)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Xu[:, 0], y=Xu[:, 1], hue=labels, s=5, palette='tab10')
    plt.title('UMAP 2D of embeddings')
    plt.legend(loc='best')
    plt.savefig(out / 'umap_2d.png', dpi=150)
    plt.close()

    print('Clustering with k-means')
    kmeans = KMeans(n_clusters=2, random_state=42).fit(Xp)
    sil = silhouette_score(Xp, kmeans.labels_)
    print('Silhouette score (k=2):', sil)
    with open(out / 'metrics.txt', 'w') as f:
        f.write(f'silhouette_k2={sil}\n')

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Xu[:, 0], y=Xu[:, 1], hue=kmeans.labels_, s=5, palette='viridis')
    plt.title('UMAP colored by k-means labels')
    plt.savefig(out / 'umap_kmeans.png', dpi=150)
    plt.close()

    print('Saved outputs to', out)


if __name__ == '__main__':
    main()
