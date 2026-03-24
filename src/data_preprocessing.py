import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path, encoding="ISO-8859-1")

    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "")

    # Remove missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove cancelled invoices
    df = df[~df['Invoice'].astype(str).str.startswith('C')]

    # Remove invalid values
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]

    # Convert date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['Price']

    df.to_csv(output_path, index=False)

    print("Data preprocessing completed.")