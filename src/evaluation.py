import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def evaluate_model(input_path, cluster_column):
    df = pd.read_csv(input_path, index_col=0)

    features = df[['Recency', 'Frequency', 'Monetary']]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    labels = df[cluster_column]

    silhouette = silhouette_score(scaled, labels)
    db_index = davies_bouldin_score(scaled, labels)

    print(f"Silhouette Score: {silhouette}")
    print(f"Davies-Bouldin Index: {db_index}")

    return silhouette, db_index