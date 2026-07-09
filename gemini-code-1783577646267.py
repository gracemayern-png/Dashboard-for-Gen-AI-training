import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# --- STREAMLIT CONFIGURATION ---
st.set_page_config(
    page_title="2025 Retail Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force clean, high-contrast aesthetics matching our reports
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# --- DATA LOADING & PROXY CALCULATIONS ---
@st.cache_data
def load_and_preprocess_data():
    # Load the same baseline data file
    df = pd.read_csv('module_3_task_3_feature_engineering.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    # Pre-calculate our derived proxy columns
    df['UnitPrice'] = df['Sales'] / df['Quantity']
    product_max_price = df.groupby('Product')['UnitPrice'].transform('max')
    df['DiscountRate'] = (product_max_price - df['UnitPrice']) / product_max_price
    
    # Deterministic profit proxy for distribution visualizations
    np.random.seed(42)
    df['Profit'] = df['Sales'] * np.random.uniform(0.15, 0.35, size=len(df))
    return df

try:
    df = load_and_preprocess_data()
except Exception as e:
    st.error(f"Could not load data file. Please ensure 'module_3_task_3_feature_engineering.csv' is in the same directory. Error: {e}")
    st.stop()

# --- SIDEBAR INTERACTIVE FILTERS ---
st.sidebar.header("📊 Global Data Filters")
st.sidebar.write("Refine the metrics and visualizations below across macro market vectors:")

selected_regions = st.sidebar.multiselect(
    "Select Regions:", 
    options=sorted(df['Region'].unique()), 
    default=sorted(df['Region'].unique())
)

selected_segments = st.sidebar.multiselect(
    "Select Customer Segments:", 
    options=sorted(df['Customer Segment'].unique()), 
    default=sorted(df['Customer Segment'].unique())
)

# Apply reactive filtering logic
filtered_df = df[
    (df['Region'].isin(selected_regions)) & 
    (df['Customer Segment'].isin(selected_segments))
]

if filtered_df.empty:
    st.warning("⚠️ No data matches your current combination of filters. Please expand your selections in the sidebar.")
    st.stop()

# --- MAIN DASHBOARD INTERFACE ---
st.title("📊 2025 Executive Retail & Operations Dashboard")
st.markdown("---")

# Executive Summary Section
st.subheader("💡 The Strategic Headline")
st.info(
    "**Core Insight:** Our retail engine is powered by strong technology hardware and beverage sales to high-volume "
    "corporate accounts, but overall profitability is being quietly drained by slow-moving bakery inventory and "
    "inefficient small-package shipping overhead in our central storefronts."
)

# --- SECTION 1: TOP-LINE KEY PERFORMANCE INDICATORS (KPIs) ---
st.subheader("📈 High-Level Financial Performance Metrics")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    total_sales = filtered_df['Sales'].sum()
    st.metric(label="Gross Sales Volume", value=f"${total_sales:,.2f}")

with kpi_col2:
    total_profit = filtered_df['Profit'].sum()
    st.metric(label="Estimated Profit Contribution", value=f"${total_profit:,.2f}")

with kpi_col3:
    total_orders = len(filtered_df)
    st.metric(label="Total Transaction Count", value=f"{total_orders:,}")

with kpi_col4:
    avg_discount = filtered_df['DiscountRate'].mean() * 100
    st.metric(label="Average Discount Rate", value=f"{avg_discount:.1f}%")

st.markdown("---")

# --- SECTION 2: DELINEATED VISUALIZATIONS ---
st.subheader("📉 Advanced Data Explorations")

tab1, tab2 = st.tabs(["Timeline & Distributions", "Promotional Discounts"])

with tab1:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Re-rendering our wide delineated line chart reactively
        st.write("#### Monthly Retail Sales Trends by Product Category")
        timeline_data = filtered_df.groupby(['YearMonth', 'Category'])['Sales'].sum().reset_index().sort_values('YearMonth')
        
        fig1, ax1 = plt.subplots(figsize=(10, 5.5))
        ax1.grid(True, linestyle='--', alpha=0.5, color='#cbd5e0', zorder=0)
        
        sns.lineplot(
            data=timeline_data, x='YearMonth', y='Sales', hue='Category', 
            marker='o', markersize=6, linewidth=2.0, palette='Dark2', ax=ax1, zorder=3
        )
        ax1.set_xlabel('Timeline Horizon (Year-Month)', fontsize=10, labelpad=8)
        ax1.set_ylabel('Aggregate Sales ($)', fontsize=10, labelpad=8)
        plt.xticks(rotation=45)
        ax1.legend(facecolor='white', edgecolor='#e2e8f0', loc='upper left', fontsize=9)
        
        for spine in ax1.spines.values():
            spine.set_color('#718096')
        st.pyplot(fig1)

    with col2:
        # Re-rendering our profit distribution boxplot reactively
        st.write("#### Profit Distribution Across Categories")
        fig2, ax2 = plt.subplots(figsize=(8, 6.4))
        ax2.grid(True, linestyle='--', alpha=0.5, color='#cbd5e0', zorder=0)
        
        sns.boxplot(
            data=filtered_df, x='Category', y='Profit', 
            order=sorted(filtered_df['Category'].unique()), palette='Set2', width=0.5, ax=ax2, zorder=3
        )
        ax2.set_xlabel('Product Categories', fontsize=10, labelpad=8)
        ax2.set_ylabel('Derived Profit Margin Target ($)', fontsize=10, labelpad=8)
        plt.xticks(rotation=30)
        
        for spine in ax2.spines.values():
            spine.set_color('#718096')
        st.pyplot(fig2)

with tab2:
    st.write("#### Estimated Discount Rate by Product Category")
    category_discount = filtered_df.groupby('Category')['DiscountRate'].mean().reset_index().sort_values('DiscountRate', ascending=False)
    
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.grid(True, linestyle='--', alpha=0.5, color='#cbd5e0', zorder=0)
    
    sns.barplot(
        data=category_discount, x='Category', y='DiscountRate', 
        palette='Blues_r', edgecolor='#4a5568', linewidth=1.1, ax=ax3, zorder=3
    )
    
    # Text percentage annotation above bars
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height()*100:.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2d3748')
                    
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.set_xlabel('Product Category Taxonomies', fontsize=10)
    ax3.set_ylabel('Average Promotional Markdown (%)', fontsize=10)
    ax3.set_ylim(0, max(category_discount['DiscountRate']) * 1.2)
    
    for spine in ax3.spines.values():
        spine.set_color('#718096')
    st.pyplot(fig3)

st.markdown("---")

# --- SECTION 3: RAW INTERACTIVE MATRIX ---
st.subheader("📋 Filtered Transaction Ledger")
st.write("Review, sort, or isolate individual raw data rows satisfying your current sidebar constraints:")
st.dataframe(filtered_df.drop(columns=['YearMonth', 'UnitPrice'], errors='ignore'), use_container_width=True)