from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_kmeans(input_path, output_path, n_clusters=4):
    df = pd.read_csv(input_path, index_col=0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['KMeans_Cluster'] = model.fit_predict(scaled)

    df.to_csv(output_path)

    print("KMeans clustering completed.")