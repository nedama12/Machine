"""Evaluation summaries and supplied cluster interpretations for Activity 3."""

from kmeans_clustering import cluster_summary, centroids_original, silhouette


CLUSTER_INTERPRETATIONS = {
    0: (
        "Moderate Digital Activity",
        "This group represents an intermediate pattern across social media hours, screen time, posts, and messages.",
    ),
    1: (
        "Low Digital Activity",
        "This group has the lowest social media hours, screen time, weekly posts, and daily messages.",
    ),
    2: (
        "High Digital Activity",
        "This group has the highest social media hours, screen time, weekly posts, and daily messages.",
    ),
}

MANUAL_EXERCISE = {
    "record_count": 100,
    "variables": ["Daily Social Media Hours", "Daily Screen Time Hours"],
    "iterations": [
        {"name": "Initial Centroids", "centroids": [(2.0, 3.5), (4.5, 6.5), (7.0, 9.5)], "sse": 168.79},
        {"name": "Iteration 1", "centroids": [(2.0613, 3.8516), (4.2729, 6.4919), (7.0406, 9.4656)], "sse": 161.24},
        {"name": "Iteration 2", "centroids": [(2.1273, 3.9303), (4.2030, 6.5273), (7.0118, 9.3353)], "sse": 159.74},
        {"name": "Iteration 3", "centroids": [(2.1273, 3.9303), (4.1625, 6.4781), (6.9686, 9.3000)], "sse": 159.22},
    ],
}


def get_cluster_distribution():
    """Return cluster membership counts as plain Python integers."""
    return {int(cluster): int(count) for cluster, count in cluster_summary.items()}


def get_cluster_profiles():
    """Build profiles from the fitted centroids and attach observed counts."""
    counts = get_cluster_distribution()
    profiles = []
    for cluster, center in enumerate(centroids_original):
        interpretation, summary = CLUSTER_INTERPRETATIONS[cluster]
        profiles.append({
            "cluster": cluster,
            "age": float(center[0]),
            "social_hours": float(center[1]),
            "screen_hours": float(center[2]),
            "posts_per_week": float(center[3]),
            "messages_per_day": float(center[4]),
            "interpretation": interpretation,
            "summary": summary,
            "count": counts.get(cluster, 0),
        })
    return profiles


def get_silhouette_score():
    """Return the silhouette score calculated during model fitting."""
    return float(silhouette)


def get_silhouette_interpretation():
    """Explain the reported score without overstating cluster separation."""
    score = get_silhouette_score()
    return (
        f"A score of {score:.4f} indicates moderate separation between the groups. "
        "Some observations from different clusters remain relatively close."
    )
