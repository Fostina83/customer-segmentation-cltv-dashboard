# data_cleaning.py

import pandas as pd

# Step 1: Load the dataset
file_path = "marketing_campaign.csv"
df = pd.read_csv(file_path, sep="\t")

print("Initial shape:", df.shape)
print("Columns:", df.columns)

# Step 2: Drop rows with critical nulls
df = df.dropna(subset=['Income', 'Dt_Customer'])

# Step 3: Fill remaining nulls
df['Income'] = df['Income'].fillna(df['Income'].median())

# Step 4: Convert date to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format="%d-%m-%Y")

# Step 5: Drop duplicates
df = df.drop_duplicates()

# Step 6: Create total spend feature
spend_cols = [col for col in df.columns if col.startswith("Mnt")]
df['Total_Spend'] = df[spend_cols].sum(axis=1)

# Step 7: Create Recency
snapshot_date = df['Dt_Customer'].max() + pd.Timedelta(days=1)
df['Recency'] = (snapshot_date - df['Dt_Customer']).dt.days

# Step 8: Create Frequency (approximate)
df['Frequency'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases']

# Step 9: Save cleaned data
df.to_csv("cleaned_marketing_campaign.csv", index=False)
print("Cleaned file saved as 'cleaned_marketing_campaign.csv'.")

# Step 10: Show preview
print(df[['ID', 'Recency', 'Frequency', 'Total_Spend']].head())
