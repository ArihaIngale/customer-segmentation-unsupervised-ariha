from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_dbscan(input_path, output_path, eps=0.8, min_samples=5):
    df = pd.read_csv(input_path, index_col=0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN_Cluster'] = model.fit_predict(scaled)

    df.to_csv(output_path)

    print("DBSCAN clustering completed.")