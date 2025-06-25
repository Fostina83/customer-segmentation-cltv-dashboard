# final_segmentation.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load cleaned data
df = pd.read_csv("cleaned_marketing_campaign.csv")

# Step 1: Select RFM features
rfm = df[['Recency', 'Frequency', 'Total_Spend']]

# Step 2: Scale features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Step 3: Apply KMeans with optimal K (replace 4 with your elbow result)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Step 4: Analyze cluster centers
cluster_summary = df.groupby('Cluster')[['Recency', 'Frequency', 'Total_Spend']].mean().round(1)
print("Cluster Summary (mean RFM):\n", cluster_summary)

# Step 5: Optional - Manually name clusters based on characteristics
# (Adjust based on your actual summary stats)
segment_map = {
    0: 'High Value',
    1: 'At Risk',
    2: 'Potential',
    3: 'Low Value'
}
df['Segment_Label'] = df['Cluster'].map(segment_map)

# Step 6: Save final segmented data
df.to_csv("segmented_customers.csv", index=False)
print("Segmented customer data saved as segmented_customers.csv")

# Step 7: Visualize segment profiles
plt.figure(figsize=(10, 6))
sns.boxplot(x='Segment_Label', y='Total_Spend', data=df)
plt.title('Customer Segments by Total Spend')
plt.show()

sns.boxplot(x='Segment_Label', y='Recency', data=df)
plt.title('Customer Segments by Recency')
plt.show()

sns.boxplot(x='Segment_Label', y='Frequency', data=df)
plt.title('Customer Segments by Frequency')
plt.show()

# Optional pairplot
sns.pairplot(df[['Recency', 'Frequency', 'Total_Spend', 'Segment_Label']], hue='Segment_Label')
plt.show()

