import pandas as pd
import numpy as np

def create_rfm(input_path, output_path):
    df = pd.read_csv(input_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Remove extreme outliers
    rfm = rfm[rfm['Monetary'] > 0]
    rfm = rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.99)]

    # Log transform
    rfm = np.log1p(rfm)

    rfm.to_csv(output_path)

    print("RFM feature engineering completed.")