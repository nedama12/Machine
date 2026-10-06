import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Load dataset
df = pd.read_csv("social_media_digital_behavior_1200.csv")


# Variables used for clustering
feature_columns = [
    "Age",
    "Daily_Social_Media_Hours",
    "Daily_Screen_Time_Hours",
    "Posts_Per_Week",
    "Messages_Per_Day"
]


# Select numerical variables
X = df[feature_columns]


# Standardize the variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# K-Means configuration
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
    max_iter=300
)


# Train the K-Means model
cluster_labels = model.fit_predict(X_scaled)


# Add cluster assignment to the dataset
df_results = df.copy()
df_results["Cluster"] = cluster_labels


# Centroids in standardized scale
centroids_scaled = model.cluster_centers_


# Centroids converted back to original units
centroids_original = scaler.inverse_transform(centroids_scaled)


# Silhouette Score
silhouette = silhouette_score(X_scaled, cluster_labels)


# Cluster summary
cluster_summary = (
    df_results["Cluster"]
    .value_counts()
    .sort_index()
)


def predict_cluster(
    age,
    daily_social_media_hours,
    daily_screen_time_hours,
    posts_per_week,
    messages_per_day
):
    new_data = pd.DataFrame({
        "Age": [age],
        "Daily_Social_Media_Hours": [daily_social_media_hours],
        "Daily_Screen_Time_Hours": [daily_screen_time_hours],
        "Posts_Per_Week": [posts_per_week],
        "Messages_Per_Day": [messages_per_day]
    })

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)

    return int(prediction[0])