from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_gmm(input_path, output_path, n_components=4):
    df = pd.read_csv(input_path, index_col=0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    model = GaussianMixture(n_components=n_components, random_state=42)
    df['GMM_Cluster'] = model.fit_predict(scaled)

    df.to_csv(output_path)

    print("GMM clustering completed.")