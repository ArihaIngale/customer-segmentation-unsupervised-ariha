from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_hierarchical(input_path, output_path, n_clusters=4):
    df = pd.read_csv(input_path, index_col=0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    df['Hierarchical_Cluster'] = model.fit_predict(scaled)

    df.to_csv(output_path)

    print("Hierarchical clustering completed.")