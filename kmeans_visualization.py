import matplotlib.pyplot as plt

from kmeans_clustering import (
    X,
    cluster_labels,
    centroids_original
)


def create_kmeans_graph():
    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        X["Daily_Social_Media_Hours"],
        X["Daily_Screen_Time_Hours"],
        c=cluster_labels,
        cmap="viridis",
        s=25,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.3
    )

    plt.scatter(
        centroids_original[:, 1],
        centroids_original[:, 2],
        marker="X",
        s=250,
        c="red",
        edgecolors="black",
        linewidths=1.5,
        label="Centroids"
    )

    plt.title(
        "K-Means Clustering: Social Media Usage and Digital Behavior"
    )

    plt.xlabel("Daily Social Media Hours")
    plt.ylabel("Daily Screen Time Hours")

    plt.colorbar(
        scatter,
        label="Cluster"
    )

    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        "static/kmeans_clusters.png",
        dpi=150
    )

    plt.close()