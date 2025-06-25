import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("segmented_customers.csv")
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['MonthYear'] = df['Dt_Customer'].dt.to_period('M').astype(str)

# Unified Custom CSS for Light Gray Theme and Black Text
st.markdown("""
    <style>
        .stApp {
            background-color: #C5C6C7;
            color: black !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, div {
            color: black !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Metric value and label text */
        div[data-testid="metric-container"] {
            color: black !important;
        }

        div[data-testid="metric-container"] > label,
        div[data-testid="stMetricDelta"] {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# ----- Header -----
st.markdown("<h1 style='text-align: center;'>Customer Segmentation & CLTV Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Insights on customer behavior, purchasing trends, and value segments</p>", unsafe_allow_html=True)
st.markdown("---")

# ----- KPIs -----
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Avg CLTV", f"${round(df['Total_Spend'].mean(), 2)}")
col2.metric("Avg Frequency", round(df['Frequency'].mean(), 2))
col3.metric("Avg Recency", round(df['Recency'].mean(), 2))

high_val_pct = df[df['Segment_Label'] == 'High Value'].shape[0] / df.shape[0] * 100
col4.metric("High Value %", f"{high_val_pct:.2f}%")
col5.metric("Segments", df['Segment_Label'].nunique())

st.markdown("---")

# ----- Segment Overview -----
st.subheader("📊 Segment Overview")
col6, col7 = st.columns(2)

with col6:
    segment_counts = df['Segment_Label'].value_counts().reset_index()
    segment_counts.columns = ['Segment_Label', 'count']
    fig1 = px.bar(segment_counts, x='Segment_Label', y='count',
                  labels={'Segment_Label': 'Segment', 'count': 'Count'},
                  title='Customers by Segment',
                  color='Segment_Label',
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1, use_container_width=True)

with col7:
    fig2 = px.pie(df, values='Total_Spend', names='Segment_Label',
                  title='CLTV Share by Segment',
                  color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig2, use_container_width=True)

# ----- Monthly Spend Trend -----
st.subheader("📈 Monthly Total Spend")
monthly = df.groupby('MonthYear')['Total_Spend'].sum().reset_index()
fig3 = px.line(monthly, x='MonthYear', y='Total_Spend', markers=True,
               title='Monthly Spend Trend', line_shape="spline")
st.plotly_chart(fig3, use_container_width=True)

# ----- Demographics -----
st.subheader("👥 Demographics")
col8, col9 = st.columns(2)

with col8:
    fig4 = px.pie(df, names='Education', title='Education Distribution',
                  hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig4, use_container_width=True)

with col9:
    fig5 = px.pie(df, names='Marital_Status', title='Marital Status',
                  hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig5, use_container_width=True)

# ----- Top Customers -----
st.subheader("🏅 Top 10 Customers by CLTV")
top_customers = df.sort_values(by='Total_Spend', ascending=False).head(10)
st.dataframe(top_customers[['ID', 'Income', 'Total_Spend', 'Segment_Label']])

# ----- Footer -----
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9em;'>Created by <strong>Fostina Bandya</strong> • Powered by Streamlit</p>", unsafe_allow_html=True)
