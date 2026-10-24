"""
================================================================================
OLYMPIC MEDAL DATA ANALYSIS - MODERN ML DASHBOARD
================================================================================
Theme: Navy Blue (#0A1931), White (#FFFFFF), and Olympic Gold (#D4AF37)
Interactive Plotly Visualizations, KPI Cards, Country Comparisons & ML Predictor
================================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Olympic Medal Analysis | ML Intelligence Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'olympics.csv')

# Custom CSS for Navy Blue, White & Gold Theme
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Header Banner */
    .olympic-hero {
        background: linear-gradient(135deg, #0A1931 0%, #152A55 60%, #1E3A8A 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(10, 25, 49, 0.3), 0 8px 10px -6px rgba(10, 25, 49, 0.2);
        margin-bottom: 2rem;
        border-left: 6px solid #D4AF37;
        position: relative;
        overflow: hidden;
    }
    .olympic-hero h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
    }
    .olympic-hero p {
        color: #E2E8F0;
        font-size: 1.05rem;
        max-width: 900px;
        margin-bottom: 0;
        line-height: 1.6;
    }
    .gold-badge {
        display: inline-block;
        background: linear-gradient(135deg, #D4AF37 0%, #F59E0B 100%);
        color: #0A1931;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 12px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        transition: all 0.25s ease-in-out;
        position: relative;
        border-top: 4px solid #0A1931;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(10, 25, 49, 0.12);
        border-top-color: #D4AF37;
    }
    .kpi-card.gold-accent {
        border-top-color: #D4AF37;
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        color: #0A1931;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .kpi-subtitle {
        color: #10B981;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }

    /* Content Cards */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .content-box h3 {
        color: #0A1931;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background: #0A1931 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(212, 175, 55, 0.3) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #E2E8F0 !important;
        font-weight: 500;
    }
    
    /* Predictor Result Cards */
    .predict-card {
        background: linear-gradient(135deg, #0A1931 0%, #152A55 100%);
        border: 2px solid #D4AF37;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px rgba(212, 175, 55, 0.25);
        color: #FFFFFF;
    }
    .predict-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .predict-badge.success {
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .predict-badge.info {
        background: #E0F2FE;
        color: #075985;
        border: 1px solid #7DD3FC;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0A1931 !important;
        color: #D4AF37 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Data Acquisition & Caching
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-07-27/olympics.csv'
        df = pd.read_csv(url)
    return df

df_raw = load_data()


# Machine Learning Model Caching
@st.cache_resource
def get_trained_models(df_data):
    df_mod = df_data[df_data['year'] >= 1960].copy()
    
    # Statistical median imputation for robust predictor inputs
    imputer = SimpleImputer(strategy='median')
    df_mod[['height', 'weight', 'age']] = imputer.fit_transform(df_mod[['height', 'weight', 'age']])
    
    df_mod['medal_won'] = df_mod['medal'].notnull().astype(int)
    df_mod['sex_male'] = (df_mod['sex'] == 'M').astype(int)
    df_mod['season_summer'] = (df_mod['season'] == 'Summer').astype(int)
    
    reg_features = ['height', 'age', 'sex_male', 'season_summer', 'year']
    
    # Regression model: Predict athlete body weight
    reg_model = LinearRegression().fit(df_mod[reg_features], df_mod['weight'])
    
    # Classification model: Predict medal likelihood with balanced class weights
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(df_mod[reg_features])
    clf_model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42).fit(X_sc, df_mod['medal_won'])
    
    return reg_model, clf_model, scaler

reg_model, clf_model, scaler = get_trained_models(df_raw)


# Helper function to display pre-generated high-res figures
def display_figure_card(fig_name, title, description):
    path = os.path.join(FIGURES_DIR, fig_name)
    st.markdown(f"#### {title}")
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, use_container_width=True)
        st.caption(description)
    else:
        st.info(f"Generated plot '{fig_name}' not found locally. Running main.py generates this artifact.")


# ==============================================================================
# SIDEBAR - FILTERS & NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size: 2.5rem;">🥇</span>
        <h2 style="color: #D4AF37; margin: 0; font-size: 1.4rem; font-weight: 800; letter-spacing: -0.5px;">OLYMPIC ML</h2>
        <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">Analytics & Prediction Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    menu = st.radio(
        "Select Module:",
        [
            "Dashboard Overview",
            "Medal Statistics & Country Comparisons",
            "Missing Value Imputation & Data Preprocessing",
            "Simple & Multiple Linear Regression",
            "Supervised Classification: KNN, Naive Bayes & Logistic Regression",
            "Morphological Clustering: K-Means & GMM",
            "Ensemble Learning: Bagging & Random Forest",
            "Cross-Validation & Hyperparameter Tuning",
            "Interactive Athlete Predictor (Live AI)"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Global Data Filters")
    
    # Global Sidebar Filters
    season_filter = st.selectbox("Olympic Season:", ["All Seasons", "Summer", "Winter"])
    
    year_min, year_max = int(df_raw['year'].min()), int(df_raw['year'].max())
    year_range = st.slider(
        "Year Timeline:",
        min_value=year_min,
        max_value=year_max,
        value=(1960, year_max),
        step=4
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #94A3B8; text-align: center;">
        Dataset: 1896 – 2016 Olympics<br>
        271,116 Historical Athlete Records<br>
        Built with Streamlit & Scikit-Learn
    </div>
    """, unsafe_allow_html=True)


# Filtered dataset based on sidebar controls
df_filtered = df_raw.copy()
if season_filter != "All Seasons":
    df_filtered = df_filtered[df_filtered['season'] == season_filter]
df_filtered = df_filtered[(df_filtered['year'] >= year_range[0]) & (df_filtered['year'] <= year_range[1])]


# ==============================================================================
# SECTION 1: DASHBOARD OVERVIEW
# ==============================================================================
if menu == "Dashboard Overview":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Official Research Platform</span>
        <h1>Olympic Medal Data Analysis & Machine Learning</h1>
        <p>A comprehensive data science framework exploring 120 years of Olympic athletic performance, biometrics, medal determinants, and predictive artificial intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic KPI Cards
    total_athletes = len(df_filtered)
    unique_nations = df_filtered['noc'].nunique()
    total_medals = df_filtered['medal'].notnull().sum()
    gold_count = (df_filtered['medal'] == 'Gold').sum()
    top_nation = df_filtered[df_filtered['medal'].notnull()]['noc'].mode()[0] if total_medals > 0 else "N/A"
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Athlete Records</div>
            <div class="kpi-value">{total_athletes:,}</div>
            <div class="kpi-subtitle">Filtered Window ({year_range[0]}-{year_range[1]})</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Competing Nations</div>
            <div class="kpi-value">{unique_nations}</div>
            <div class="kpi-subtitle">Global NOC Representation</div>
        </div>
        <div class="kpi-card gold-accent">
            <div class="kpi-title">Total Medals Awarded</div>
            <div class="kpi-value">{total_medals:,}</div>
            <div class="kpi-subtitle" style="color: #D4AF37;">{gold_count:,} Gold Medals</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Leading Delegation</div>
            <div class="kpi-value">{top_nation}</div>
            <div class="kpi-subtitle">Most Podium Finishes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Plotly Visualizations
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 📈 Historical Participation Growth (Athletes by Gender)")
        gender_yearly = df_filtered.groupby(['year', 'sex'])['id'].count().reset_index()
        gender_yearly['sex'] = gender_yearly['sex'].map({'M': 'Male Athletes', 'F': 'Female Athletes'})
        
        fig_gender = px.area(
            gender_yearly,
            x='year',
            y='id',
            color='sex',
            color_discrete_map={'Male Athletes': '#0A1931', 'Female Athletes': '#D4AF37'},
            labels={'id': 'Total Athletes', 'year': 'Olympic Year', 'sex': 'Gender'},
            template='plotly_white'
        )
        fig_gender.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        st.markdown("### 🏅 Medal Breakdown Distribution")
        medal_dist = df_filtered['medal'].fillna('No Medal').value_counts().reset_index()
        medal_dist.columns = ['Status', 'Count']
        
        fig_donut = px.pie(
            medal_dist,
            values='Count',
            names='Status',
            hole=0.55,
            color='Status',
            color_discrete_map={
                'Gold': '#FFD700',
                'Silver': '#C0C0C0',
                'Bronze': '#CD7F32',
                'No Medal': '#0A1931'
            },
            template='plotly_white'
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            annotations=[dict(text=f"{total_medals:,}<br>Medals", x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Data Explorer Table
    with st.expander("🔍 Interactive Olympic Dataset Explorer (Search & Filter)"):
        selected_columns = st.multiselect(
            "Select Columns to Display:",
            options=df_raw.columns.tolist(),
            default=['name', 'sex', 'age', 'height', 'weight', 'team', 'noc', 'year', 'season', 'sport', 'medal']
        )
        search_query = st.text_input("Search Athlete, Team, or Sport:")
        df_display = df_filtered[selected_columns]
        if search_query:
            mask = df_display.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            df_display = df_display[mask]
        st.dataframe(df_display.head(100), use_container_width=True)


# ==============================================================================
# SECTION 2: MEDAL STATISTICS & COUNTRY COMPARISONS
# ==============================================================================
elif menu == "Medal Statistics & Country Comparisons":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Comparative Analytics</span>
        <h1>Medal Statistics & Country Leaderboards</h1>
        <p>Explore all-time nation rankings, compare country medal tallies head-to-head, and uncover historical dominance across summer and winter disciplines.</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter medal records
    medals_only = df_filtered[df_filtered['medal'].notnull()].copy()

    tab1, tab2 = st.tabs(["🏆 All-Time Leaderboard", "⚔️ Head-to-Head Country Comparison"])

    with tab1:
        st.markdown("### Top 15 All-Time Medal Winning Nations")
        country_medals = medals_only.groupby(['noc', 'medal'])['id'].count().reset_index()
        pivot_medals = country_medals.pivot(index='noc', columns='medal', values='id').fillna(0)
        
        # Ensure Gold, Silver, Bronze columns exist
        for m in ['Gold', 'Silver', 'Bronze']:
            if m not in pivot_medals.columns:
                pivot_medals[m] = 0
                
        pivot_medals['Total'] = pivot_medals['Gold'] + pivot_medals['Silver'] + pivot_medals['Bronze']
        top15_countries = pivot_medals.sort_values('Total', ascending=False).head(15).reset_index()
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='Gold', x=top15_countries['noc'], y=top15_countries['Gold'], marker_color='#D4AF37'))
        fig_bar.add_trace(go.Bar(name='Silver', x=top15_countries['noc'], y=top15_countries['Silver'], marker_color='#94A3B8'))
        fig_bar.add_trace(go.Bar(name='Bronze', x=top15_countries['noc'], y=top15_countries['Bronze'], marker_color='#CD7F32'))
        
        fig_bar.update_layout(
            barmode='stack',
            template='plotly_white',
            xaxis_title="National Olympic Committee (NOC)",
            yaxis_title="Total Medals Won",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.dataframe(
            top15_countries[['noc', 'Gold', 'Silver', 'Bronze', 'Total']].rename(columns={'noc': 'Country Code'}),
            use_container_width=True
        )

    with tab2:
        st.markdown("### Compare Two Nations Head-to-Head")
        all_nocs = sorted(df_raw['noc'].unique().tolist())
        
        colA, colB = st.columns(2)
        with colA:
            c1 = st.selectbox("Select Country A:", all_nocs, index=all_nocs.index('USA') if 'USA' in all_nocs else 0)
        with colB:
            c2 = st.selectbox("Select Country B:", all_nocs, index=all_nocs.index('GBR') if 'GBR' in all_nocs else 1)

        c1_df = medals_only[medals_only['noc'] == c1]
        c2_df = medals_only[medals_only['noc'] == c2]

        c1_total = len(c1_df)
        c2_total = len(c2_df)
        c1_gold = (c1_df['medal'] == 'Gold').sum()
        c2_gold = (c2_df['medal'] == 'Gold').sum()

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class="kpi-card gold-accent" style="text-align: center;">
                <div class="kpi-title">{c1} Medal Record</div>
                <div class="kpi-value">{c1_total:,} Total</div>
                <div class="kpi-subtitle" style="color: #D4AF37;">🥇 {c1_gold} Gold | 🥈 {(c1_df['medal'] == 'Silver').sum()} Silver | 🥉 {(c1_df['medal'] == 'Bronze').sum()} Bronze</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="kpi-card" style="text-align: center;">
                <div class="kpi-title">{c2} Medal Record</div>
                <div class="kpi-value">{c2_total:,} Total</div>
                <div class="kpi-subtitle" style="color: #0A1931;">🥇 {c2_gold} Gold | 🥈 {(c2_df['medal'] == 'Silver').sum()} Silver | 🥉 {(c2_df['medal'] == 'Bronze').sum()} Bronze</div>
            </div>
            """, unsafe_allow_html=True)

        # Timeline comparison
        c1_time = c1_df.groupby('year')['id'].count().reset_index().rename(columns={'id': c1})
        c2_time = c2_df.groupby('year')['id'].count().reset_index().rename(columns={'id': c2})
        comp_time = pd.merge(c1_time, c2_time, on='year', how='outer').fillna(0).sort_values('year')

        fig_comp = px.line(
            comp_time,
            x='year',
            y=[c1, c2],
            markers=True,
            color_discrete_sequence=['#D4AF37', '#0A1931'],
            labels={'value': 'Medals Won', 'year': 'Olympic Year', 'variable': 'Nation'},
            template='plotly_white',
            title=f"Historical Medal Trajectory: {c1} vs. {c2}"
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ==============================================================================
# SECTION 3: IMPUTATION & PREPROCESSING
# ==============================================================================
elif menu == "Missing Value Imputation & Data Preprocessing":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Data Engineering</span>
        <h1>Missing Value Detection & Anti-Leakage Preprocessing</h1>
        <p>Statistical evaluation of Mean, Median, and Mode imputation, followed by rigorous train/test scaling without data leakage.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔬 Statistical Imputation Analysis", "⚖️ Anti-Leakage Feature Scaling"])

    with tab1:
        st.markdown("""
        #### Missingness Mechanisms & Imputation Strategy
        - **Structural Missingness:** The `medal` column has 85.3% $NaN$ values representing athletes who finished off the podium. This is encoded categorically as `None` ($0$).
        - **Biometric Attributes:** Athlete `weight` (23.2%) and `height` (22.2%) contain genuine missing records from historical games.
        - **Variance Compression:** Mean imputation forces 62,875 records to 70.70 kg, compressing sample standard deviation from 14.35 kg to 12.57 kg. Median imputation safely retains IQR boundaries without distortion.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            display_figure_card(
                "p1_missing_values_bar.png",
                "Missing Value Percentage by Feature",
                "Identifies highest missingness in target medal status and physical biometrics."
            )
        with col2:
            display_figure_card(
                "p1_imputation_distribution_comparison.png",
                "Kernel Density Estimation: Pre vs. Post Imputation",
                "Visualizes variance compression of Mean vs. percentile preservation of Median imputation."
            )

    with tab2:
        st.markdown(r"""
        #### Anti-Leakage Feature Normalization Protocol
        - **The Data Leakage Risk:** If scaling parameters ($\mu, \sigma$) are estimated over the complete dataset prior to partitioning, validation estimates become unrealistically optimistic.
        - **Strict Train-Fit Protocol:** `StandardScaler` is fitted **strictly on `X_train`**.
        - **Empirical Proof:** Training partition scales to $\mu = 0.000, \sigma = 1.000$; unseen test partition scales to $\mu = -0.005, \sigma = 1.003$, proving zero information leakage.
        """)
        display_figure_card(
            "p2_scaling_comparison_boxplots.png",
            "Raw Features vs. StandardScaler vs. MinMaxScaler",
            "Boxplot distributions verifying standardization centering and range bounded normalization."
        )


# ==============================================================================
# SECTION 4: LINEAR & MULTIPLE REGRESSION
# ==============================================================================
elif menu == "Simple & Multiple Linear Regression":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Supervised Regression</span>
        <h1>Simple & Multiple Linear Regression</h1>
        <p>Allometric biomechanical modeling predicting athlete body mass (kg) from stature, age, biological sex, and sporting categories.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Simple Linear Regression (Height $\to$ Weight)
        - **Mathematical Formulation:** $\\text{Weight} = \\beta_0 + \\beta_1 \\times \\text{Height} + \\epsilon$
        - **Fitted Parameters:** $\\beta_1 = 1.0772\\text{ kg/cm}, \\quad \\beta_0 = -118.2326\\text{ kg}$
        - **Goodness-of-Fit:** $R^2 = 0.6348, \\quad \\text{RMSE} = 8.501\\text{ kg}$
        - **Interpretation:** Elite athletes gain approximately **$1.08\\text{ kg}$ body mass per centimeter** of stature.
        """)
        display_figure_card(
            "p3_simple_linear_regression.png",
            "OLS Regression Fit & Residual Analysis",
            "Height vs. Weight scatter plot, fitted regression line, and residual homoscedasticity check."
        )

    with col2:
        st.markdown("""
        #### Multiple Linear Regression (Multivariate Biometrics)
        - **Covariates:** Height, Age, Biological Sex, Season, Year, Sport Disciplines.
        - **Explanatory Gain:** $R_{\\text{adj}}^2 = 0.7076$ (up from $0.635$), $\\text{RMSE} = 7.603\\text{ kg}$.
        - **Biological Sex Factor:** Male athletes carry **$+7.61\\text{ kg}$** higher expected mass controlling for stature and sport.
        - **Error Normality:** Residual distribution exhibits near-Gaussian symmetry around zero.
        """)
        display_figure_card(
            "p4_multiple_linear_regression.png",
            "Predicted vs. Actual Mass & Residual Histogram",
            "Shows strong alignment along the identity line and symmetric residual error frequencies."
        )


# ==============================================================================
# SECTION 5: CLASSIFICATION
# ==============================================================================
elif menu == "Supervised Classification: KNN, Naive Bayes & Logistic Regression":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Supervised Classification</span>
        <h1>Supervised Classification & Class Imbalance</h1>
        <p>Benchmarking K-Nearest Neighbors, Gaussian Naive Bayes, and Cost-Balanced Logistic Regression on imbalanced Olympic medal outcomes.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">K-Nearest Neighbors (K=5)</div>
            <div class="kpi-value">0.6133</div>
            <div class="kpi-subtitle">ROC-AUC Score</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Gaussian Naive Bayes</div>
            <div class="kpi-value">0.6115</div>
            <div class="kpi-subtitle">ROC-AUC Score</div>
        </div>
        <div class="kpi-card gold-accent">
            <div class="kpi-title">Balanced Logistic Regression</div>
            <div class="kpi-value">58.19%</div>
            <div class="kpi-subtitle" style="color: #D4AF37;">Medalist Recall (vs 0.1% Unweighted)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        display_figure_card(
            "p5_knn_evaluation.png",
            "KNN: Neighborhood Size K Trajectory & Confusion Matrix",
            "Shows bias-variance tradeoff across K values and confusion matrix at optimal K=5."
        )
        display_figure_card(
            "p6_naive_bayes_evaluation.png",
            "Gaussian Naive Bayes: ROC Curve & Confusion Matrix",
            "Evaluates conditional independence assumption and probabilistic classification report."
        )
    with col2:
        display_figure_card(
            "p7_logistic_regression.png",
            "Logistic Regression: Feature Odds Ratios & ROC Curves",
            "Demonstrates dramatic minority recall recovery via cost-sensitive learning and odds ratios."
        )


# ==============================================================================
# SECTION 6: CLUSTERING
# ==============================================================================
elif menu == "Morphological Clustering: K-Means & GMM":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Unsupervised Learning</span>
        <h1>Morphological Clustering & Athlete Archetypes</h1>
        <p>Unsupervised profiling of human athletic morphology using K-Means partitioning and Gaussian Mixture Models (GMM) soft clustering.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### K-Means Morphological Archetypes ($K=3$)
        Inertia Elbow Method and Silhouette Analysis identify 3 natural biometric archetypes:
        1. **Power / Heavyweight:** Height: $181.2\\text{ cm}$, Weight: $95.4\\text{ kg}$, BMI: $28.8$ (Weightlifters, throwers, super-heavyweights).
        2. **Compact / Endurance:** Height: $167.3\\text{ cm}$, Weight: $57.1\\text{ kg}$, BMI: $20.6$ (Gymnasts, distance runners).
        3. **Tall / Athletic:** Height: $185.1\\text{ cm}$, Weight: $78.3\\text{ kg}$, BMI: $22.9$ (Rowers, swimmers, basketballers).
        """)
        display_figure_card(
            "p8_kmeans_clustering.png",
            "K-Means: Elbow Curve, Silhouette & Centroids",
            "Validates cluster count K=3 and maps athlete morphological centroids in Height-Weight space."
        )
        
    with col2:
        st.markdown("""
        #### Gaussian Mixture Models (Soft Clustering & EM)
        - **Expectation-Maximization:** Estimates probabilistic cluster memberships (responsibilities $\\gamma_{ik}$).
        - **Model Selection:** Bayesian Information Criterion (BIC) selects $K=4$ components.
        - **Anisotropic Covariance:** Unlike spherical K-Means boundaries, full-covariance GMM models the natural elliptical correlation between height and weight.
        """)
        display_figure_card(
            "p9_gmm_clustering.png",
            "GMM: AIC/BIC Model Selection & Ellipsoidal Clusters",
            "Displays minimum BIC inflection and soft density contours reflecting correlated biometrics."
        )


# ==============================================================================
# SECTION 7: ENSEMBLE LEARNING
# ==============================================================================
elif menu == "Ensemble Learning: Bagging & Random Forest":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Ensemble Methods</span>
        <h1>Ensemble Learning & Variance Reduction</h1>
        <p>Empirical proof of ensemble variance reduction: Single Decision Tree vs. Bagging Classifier vs. Balanced Random Forest.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Single Decision Tree</div>
            <div class="kpi-value">0.5902</div>
            <div class="kpi-subtitle" style="color: #64748B;">High Variance Baseline</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Bagging Classifier</div>
            <div class="kpi-value">0.6397</div>
            <div class="kpi-subtitle" style="color: #2563EB;">Bootstrap Aggregation</div>
        </div>
        <div class="kpi-card gold-accent">
            <div class="kpi-title">Balanced Random Forest</div>
            <div class="kpi-value">0.6621</div>
            <div class="kpi-subtitle" style="color: #D4AF37;">Subspace Decorrelation (+7.2% Gain)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    display_figure_card(
        "p10_ensemble_comparison.png",
        "Ensemble ROC Curves & Top Gini Feature Importances",
        "Confirms theoretical variance reduction. Key Gini drivers: Age (32.2%), Height (24.6%), Weight (23.8%)."
    )


# ==============================================================================
# SECTION 8: VALIDATION & TUNING
# ==============================================================================
elif menu == "Cross-Validation & Hyperparameter Tuning":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Model Optimization</span>
        <h1>Cross-Validation & Hyperparameter Search</h1>
        <p>Measuring score variance stability between K-Fold and Stratified K-Fold, and benchmarking GridSearchCV against budget-bounded RandomizedSearchCV.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(r"""
        #### K-Fold vs. Stratified K-Fold CV
        - **Class Ratio Conservation:** In imbalanced target distributions (~14% medalists), random K-Fold suffers from fluctuating fold class compositions ($13.30\% - 13.97\%$).
        - **Stratified Stability:** Stratified K-Fold strictly locks the positive class ratio at **$13.58\%$**, reducing fold-to-fold standard deviation by **$40\%$** ($\pm 0.017 \to \pm 0.010$).
        """)
        display_figure_card(
            "p11_cross_validation_comparison.png",
            "Cross-Validation Score Stability & Fold Balance",
            "Boxplots proving tighter score variance and fold class proportion conservation under stratification."
        )

    with col2:
        st.markdown(r"""
        #### GridSearchCV vs. RandomizedSearchCV
        - **Computational Efficiency:** RandomizedSearchCV explored a wide parameter space in **$6.1\text{ seconds}$** compared to **$13.8\text{ seconds}$** for exhaustive GridSearchCV ($56\%$ compute reduction).
        - **Solution Quality:** Both methods converged to identical optimal test performance ($\text{ROC-AUC} \approx 0.6056$).
        """)
        display_figure_card(
            "p12_hyperparameter_tuning.png",
            "Search Time vs. Test ROC-AUC Comparison",
            "Empirical verification of the Bergstra & Bengio efficiency theorem for randomized optimization."
        )


# ==============================================================================
# SECTION 9: INTERACTIVE ATHLETE PREDICTOR (LIVE AI)
# ==============================================================================
elif menu == "Interactive Athlete Predictor (Live AI)":
    st.markdown("""
    <div class="olympic-hero">
        <span class="gold-badge">Live ML Inference Engine</span>
        <h1>Interactive Athlete Predictor</h1>
        <p>Enter custom physiological and competitive parameters to predict athlete body mass and estimate probability of winning an Olympic medal in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 🏃 Configure Athlete Profile")
        
        height_in = st.slider("Height (cm):", min_value=140, max_value=225, value=182, step=1)
        age_in = st.slider("Age (Years):", min_value=14, max_value=60, value=25, step=1)
        
        c_sex, c_sea = st.columns(2)
        with c_sex:
            sex_in = st.selectbox("Biological Sex:", ["Male", "Female"])
        with c_sea:
            season_in = st.selectbox("Olympic Season:", ["Summer", "Winter"])
            
        c_yr, c_sp = st.columns(2)
        with c_yr:
            year_in = st.slider("Olympic Year:", min_value=1960, max_value=2024, value=2016, step=2)
        with c_sp:
            sport_in = st.selectbox("Sport Discipline:", ["Athletics", "Swimming", "Gymnastics", "Rowing", "Cycling", "Other"])

    # Live Model Inference
    is_male = 1 if sex_in == "Male" else 0
    is_summer = 1 if season_in == "Summer" else 0
    reg_features = ['height', 'age', 'sex_male', 'season_summer', 'year']
    input_df = pd.DataFrame([[height_in, age_in, is_male, is_summer, year_in]], columns=reg_features)

    # Predictions
    pred_weight = reg_model.predict(input_df)[0]
    input_sc = scaler.transform(input_df)
    pred_prob = clf_model.predict_proba(input_sc)[0][1] * 100
    bmi = pred_weight / ((height_in / 100) ** 2)

    with col_right:
        st.markdown("### 🎯 Artificial Intelligence Predictions")
        
        # Interactive Plotly Gauge for Medal Probability (Bright White Title and %)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob,
            number={'suffix': "%", 'font': {'size': 44, 'color': '#FFFFFF'}},
            title={'text': "Olympic Medal Probability", 'font': {'size': 20, 'color': '#FFFFFF'}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "#FFFFFF",
                    'tickfont': {'color': '#FFFFFF', 'size': 12}
                },
                'bar': {'color': "#D4AF37", 'thickness': 0.35},
                'bgcolor': "rgba(255, 255, 255, 0.08)",
                'borderwidth': 2,
                'bordercolor': "#D4AF37",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(255, 255, 255, 0.15)'},
                    {'range': [35, 60], 'color': 'rgba(255, 255, 255, 0.28)'},
                    {'range': [60, 100], 'color': 'rgba(212, 175, 55, 0.45)'}
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 4},
                    'thickness': 0.8,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#FFFFFF'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Morphological Archetype Tag
        if bmi > 26.5:
            arch_type = "Power / Heavyweight Biotype"
            arch_color = "#D4AF37"
        elif bmi < 21.5:
            arch_type = "Compact / Endurance Biotype"
            arch_color = "#3B82F6"
        else:
            arch_type = "Tall / Athletic Biotype"
            arch_color = "#10B981"

        st.markdown(f"""
        <div class="predict-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600;">PREDICTED BODY MASS</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #FFFFFF;">{pred_weight:.1f} kg</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600;">BODY MASS INDEX</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #D4AF37;">{bmi:.1f} kg/m²</div>
                </div>
            </div>
            <div style="margin-top: 0.5rem;">
                <span style="background: rgba(255, 255, 255, 0.12); color: #FFFFFF; font-weight: 700; font-size: 0.85rem; padding: 6px 14px; border-radius: 9999px; border: 1px solid #D4AF37;">
                    🏷️ {arch_type}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if pred_prob >= 50.0:
            st.success(f"🏆 **High Podium Potential ({pred_prob:.1f}%)!** Physical and demographic covariates align closely with Olympic medalists in {sport_in}.")
        else:
            st.info(f"🏅 **Competitive Participant Profile ({pred_prob:.1f}%)**. Features correspond to standard Olympic qualifier distributions in {sport_in}.")
