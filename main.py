from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_rfm
from src.clustering.kmeans import run_kmeans
from src.evaluation import evaluate_model

RAW_DATA = "data/raw/online_retail.csv"
CLEAN_DATA = "data/processed/cleaned_data.csv"
RFM_DATA = "data/processed/rfm_data.csv"
CLUSTER_DATA = "data/processed/rfm_with_clusters.csv"


def main():

    print("STEP 1: Data Preprocessing")
    preprocess_data(RAW_DATA, CLEAN_DATA)

    print("STEP 2: Feature Engineering")
    create_rfm(CLEAN_DATA, RFM_DATA)

    print("STEP 3: Running KMeans Clustering")
    run_kmeans(RFM_DATA, CLUSTER_DATA)

    print("STEP 4: Evaluating Model")
    evaluate_model(CLUSTER_DATA, "KMeans_Cluster")

    print("Pipeline Completed Successfully")


if __name__ == "__main__":
    main()