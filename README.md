# AI-Driven Customer Intelligence System for Strategic Business Decision Making

## Project Overview
This project focuses on customer segmentation using unsupervised machine learning techniques. The objective is to identify hidden patterns in customer behavior and group customers into meaningful segments without labeled data.

Customer segmentation helps businesses understand their customers better and enables data-driven decision-making for marketing strategies, customer retention, and revenue optimization.

The project implements multiple clustering algorithms and compares their performance to determine the best segmentation strategy.



# Problem Statement
Companies collect large volumes of customer data but often lack labeled information about customer behavior. Without proper segmentation, businesses cannot identify high-value customers, predict churn risk, or target marketing campaigns effectively.

This project applies unsupervised learning techniques to discover meaningful customer segments and generate actionable business insights.



# Dataset Description
The dataset contains customer purchasing and behavioral data.

Key features used in this project include:

- Recency – Days since the last purchase
- Frequency – Number of purchases made by the customer
- Monetary – Total spending by the customer

These RFM features are widely used in customer analytics and marketing segmentation.

Dataset size: 5000+ customer records.



# Project Workflow

The project follows a complete machine learning pipeline:

1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering (RFM Analysis)
4. Clustering Model Implementation
5. Model Comparison
6. Dimensionality Reduction (PCA, t-SNE)
7. Cluster Visualization
8. Business Insights & Recommendations



# Algorithms Used

The following unsupervised learning algorithms were implemented and compared:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model (GMM)

Dimensionality reduction techniques used:

- Principal Component Analysis (PCA)
- t-SNE Visualization

---

# Project Structure


customer-segmentation-unsupervised
│
├── data
│ ├── raw
│ └── processed
│
├── notebooks
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_clustering_models.ipynb
│ ├── 05_model_comparison.ipynb
│ ├── 06_visualization.ipynb
│ └── 07_business_insights.ipynb
│
├── src
│
├── results
│ ├── cluster_plots
│ ├── metrics
│ └── pca_outputs
│
├── reports
│ ├── final_report.pdf
│ └── presentation.pptx
│
├── main.py
├── README.md
└── requirements.txt


---

# How to Run the Project

Clone the repository:

git clone https://github.com/ArihaIngale/customer-segmentation-unsupervised-ariha.git


Navigate to the project directory:
cd customer-segmentation-unsupervised


Install dependencies:
pip install -r requirements.txt


Run the main pipeline:
python main.py


---


# Key Results

Optimal number of clusters identified: **4**

Best performing model: **K-Means Clustering**

Key evaluation metrics:

- Silhouette Score
- Davies-Bouldin Index
- Cluster separation using PCA visualization

---


# Sample Visualizations

The following visualizations were generated:

- Elbow Method for Optimal Cluster Selection
- PCA Cluster Visualization
- t-SNE Cluster Visualization
- Cluster comparison across algorithms

Example plots are stored in:results/cluster_plots/


---

# Business Insights

Customer segmentation revealed the following customer groups:

1. High Value Customers
   - Frequent purchases
   - High spending

2. Budget Customers
   - Moderate purchase frequency
   - Low spending

3. Occasional Buyers
   - Infrequent purchases

4. At-Risk Customers
   - Long time since last purchase

---

# Marketing Recommendations

Based on the segmentation results:

- High Value Customers → Loyalty programs and premium offers
- Budget Customers → Discount campaigns
- Occasional Buyers → Personalized product recommendations
- At-Risk Customers → Retention campaigns

---


# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
