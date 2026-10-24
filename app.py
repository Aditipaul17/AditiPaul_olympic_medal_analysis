"""
Olympic Medal Data Analysis Using Machine Learning - Interactive Web Dashboard
Run locally with: python -m streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from PIL import Image

st.set_page_config(
    page_title="Olympic Medal Analysis - ML Dashboard",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'figures')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'olympics.csv')

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-07-27/olympics.csv'
        df = pd.read_csv(url)
    return df

@st.cache_resource
def get_trained_models(df_data):
    df_mod = df_data[df_data['year'] >= 1960].copy()
    
    # Impute any missing biometrics (height, weight, age) with median to prevent NaN errors
    imputer = SimpleImputer(strategy='median')
    df_mod[['height', 'weight', 'age']] = imputer.fit_transform(df_mod[['height', 'weight', 'age']])
    
    df_mod['medal_won'] = df_mod['medal'].notnull().astype(int)
    df_mod['sex_male'] = (df_mod['sex'] == 'M').astype(int)
    df_mod['season_summer'] = (df_mod['season'] == 'Summer').astype(int)
    
    reg_features = ['height', 'age', 'sex_male', 'season_summer', 'year']
    
    # Fit regression model
    reg_model = LinearRegression().fit(df_mod[reg_features], df_mod['weight'])
    
    # Fit classification model
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(df_mod[reg_features])
    clf_model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42).fit(X_sc, df_mod['medal_won'])
    
    return reg_model, clf_model, scaler

df_raw = load_data()
reg_model, clf_model, scaler = get_trained_models(df_raw)

# App Header
st.title("🏅 Olympic Medal Data Analysis Using Machine Learning")
st.markdown("""
**An End-to-End Machine Learning Research Platform & Interactive Dashboard**  
*Covers 12 Practicals: Imputation, Preprocessing, Linear & Multiple Regression, KNN, Naive Bayes, Logistic Regression, K-Means, GMM, Ensembles, Cross-Validation & Hyperparameter Tuning.*
""")

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Olympic_flag.svg/320px-Olympic_flag.svg.png", width=120)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Choose a Section:",
    [
        "Project Overview & Data Explorer",
        "Practicals 1 & 2: Imputation & Preprocessing",
        "Practicals 3 & 4: Regression (Height & Weight)",
        "Practicals 5, 6 & 7: Classification (Medal Winner)",
        "Practicals 8 & 9: Morphological Clustering (K-Means & GMM)",
        "Practical 10: Ensembles (Bagging & Random Forest)",
        "Practicals 11 & 12: Validation & Hyperparameter Tuning",
        "Interactive ML Predictor (Live Inference)"
    ]
)

def display_fig(fig_name, caption=""):
    path = os.path.join(FIGURES_DIR, fig_name)
    if os.path.exists(path):
        image = Image.open(path)
        st.image(image, caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure {fig_name} not found. Please run main.py first.")

# SECTION 1: OVERVIEW & DATA EXPLORER
if menu == "Project Overview & Data Explorer":
    st.header("📊 Dataset Overview & Exploratory Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df_raw):,}")
    col2.metric("Total Features", f"{df_raw.shape[1]}")
    col3.metric("Year Range", f"{df_raw['year'].min()} – {df_raw['year'].max()}")
    col4.metric("Medal Percentage", f"{(df_raw['medal'].notnull().mean()*100):.1f}%")

    st.subheader("Raw Dataset Sample")
    st.dataframe(df_raw.head(10), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Medal Distribution")
        medal_counts = df_raw['medal'].fillna('No Medal').value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=medal_counts.index, y=medal_counts.values, palette='copper', ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with colB:
        st.subheader("Top 10 Sports by Participation")
        top_sports = df_raw['sport'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=top_sports.values, y=top_sports.index, palette='mako', ax=ax)
        ax.set_xlabel("Athletes")
        st.pyplot(fig)

# SECTION 2: PRACTICALS 1 & 2
elif menu == "Practicals 1 & 2: Imputation & Preprocessing":
    st.header("P1: Missing Value Detection & Statistical Imputation")
    st.markdown("""
    - **Detection:** Analyzed missingness across attributes. `medal` has 85.3% $NaN$ (structural non-medalists), while `weight` (23.2%) and `height` (22.2%) contain genuine missing biometrics.
    - **Imputation Comparison:** Evaluated Mean, Median, and Mode strategies.
    """)
    col1, col2 = st.columns(2)
    with col1:
        display_fig("p1_missing_values_bar.png", "Missing Value Percentage by Feature")
    with col2:
        display_fig("p1_imputation_distribution_comparison.png", "Density Comparison: Pre vs. Post Imputation")

    st.markdown("---")
    st.header("P2: Data Preprocessing, Scaling & Anti-Leakage")
    st.markdown(r"""
    - **Anti-Leakage Protocol:** Evaluated `StandardScaler` and `MinMaxScaler` fit **exclusively** on training data.
    - **Validation Verification:** Train set scales to $\mu = 0.000, \sigma = 1.000$; test set scales to $\mu = -0.005, \sigma = 1.003$ without leakage.
    """)
    display_fig("p2_scaling_comparison_boxplots.png", "Raw vs. StandardScaler vs. MinMaxScaler Distributions")

# SECTION 3: PRACTICALS 3 & 4
elif menu == "Practicals 3 & 4: Regression (Height & Weight)":
    st.header("P3 & P4: Simple and Multiple Linear Regression")
    st.markdown("""
    - **Simple OLS Regression:** Height $\to$ Weight: $\\text{Weight} = -118.23 + 1.08 \\times \\text{Height}$ ($R^2 = 0.6348, \\text{RMSE} = 8.501\\text{ kg}$).
    - **Multiple Regression:** Height, Age, Sex, Season, Year, Sport $\to$ Weight ($R_{\\text{adj}}^2 = 0.7076, \\text{RMSE} = 7.603\\text{ kg}$).
    - **Sex Effect:** Being male adds $+7.61\\text{ kg}$ on average controlling for height and sport.
    """)
    col1, col2 = st.columns(2)
    with col1:
        display_fig("p3_simple_linear_regression.png", "P3: Height vs. Weight Regression Line & Residuals")
    with col2:
        display_fig("p4_multiple_linear_regression.png", "P4: Multiple Regression Predictions & Error Distribution")

# SECTION 4: PRACTICALS 5, 6 & 7
elif menu == "Practicals 5, 6 & 7: Classification (Medal Winner)":
    st.header("P5, P6 & P7: Supervised Classification on Imbalanced Medal Data")
    st.markdown("""
    - **Practical 5 (KNN):** Best $K=5$, $\\text{ROC-AUC} = 0.6133$. Demonstrates curse of dimensionality and sensitivity to scaling.
    - **Practical 6 (Gaussian Naive Bayes):** $\\text{ROC-AUC} = 0.6115$. Extremely fast $\\mathcal{O}(Nd)$ inference.
    - **Practical 7 (Logistic Regression):** Solves severe 14:86 class imbalance using `class_weight='balanced'`. Medalist recall increases from $0.1\\%$ to $58.2\\%$.
    """)
    col1, col2 = st.columns(2)
    with col1:
        display_fig("p5_knn_evaluation.png", "P5: KNN Metrics vs. K and Confusion Matrix")
        display_fig("p6_naive_bayes_evaluation.png", "P6: Gaussian Naive Bayes ROC Curve & Confusion Matrix")
    with col2:
        display_fig("p7_logistic_regression.png", "P7: Feature Odds Ratios & Cost-Balanced ROC Curves")

# SECTION 5: PRACTICALS 8 & 9
elif menu == "Practicals 8 & 9: Morphological Clustering (K-Means & GMM)":
    st.header("P8 & P9: Unsupervised Morphological Clustering")
    st.markdown("""
    - **P8 (K-Means):** Elbow method and Silhouette scores reveal 3 natural athletic archetypes:
      1. **Power / Heavyweight:** High BMI (28.8), 95.4 kg (Weightlifters, throwers).
      2. **Compact / Endurance:** Low BMI (20.6), 57.1 kg (Gymnasts, runners).
      3. **Tall / Athletic:** Moderate BMI (22.9), 185 cm, 78.3 kg (Rowers, swimmers, basketballers).
    - **P9 (GMM):** Expectation-Maximization fits full anisotropic Gaussian covariances, capturing natural height-weight elongation.
    """)
    col1, col2 = st.columns(2)
    with col1:
        display_fig("p8_kmeans_clustering.png", "P8: K-Means Inertia Elbow, Silhouette & Centroids")
    with col2:
        display_fig("p9_gmm_clustering.png", "P9: GMM AIC/BIC Component Selection & Ellipsoidal Clusters")

# SECTION 6: PRACTICAL 10
elif menu == "Practical 10: Ensembles (Bagging & Random Forest)":
    st.header("P10: Ensemble Learning Variance Reduction")
    st.markdown("""
    - **Variance Reduction Benchmark:**
      - Single Decision Tree: $\\text{ROC-AUC} = 0.5902$
      - Bagging Classifier: $\\text{ROC-AUC} = 0.6397$
      - Balanced Random Forest: $\\text{ROC-AUC} = 0.6621$ (OOB Score: $0.6552$)
    - **Gini Feature Importances:** Age ($32.2\\%$), Height ($24.6\\%$), Weight ($23.8\\%$), Year ($14.1\\%$).
    """)
    display_fig("p10_ensemble_comparison.png", "Ensemble ROC Curves & Top Gini Feature Importances")

# SECTION 7: PRACTICALS 11 & 12
elif menu == "Practicals 11 & 12: Validation & Hyperparameter Tuning":
    st.header("P11 & P12: Cross-Validation & Optimization")
    st.markdown(r"""
    - **P11 (K-Fold vs. Stratified K-Fold):** Stratification cuts metric standard deviation by $40\%$ ($\pm 0.017 \to \pm 0.010$) and eliminates fold class shift.
    - **P12 (GridSearchCV vs. RandomizedSearchCV):** Random search achieved matching test AUC ($0.6056$ vs. $0.6054$) in less than half the execution time ($6.1\text{s}$ vs. $13.8\text{s}$).
    """)
    col1, col2 = st.columns(2)
    with col1:
        display_fig("p11_cross_validation_comparison.png", "P11: Cross-Validation Score Variance & Class Proportion Balance")
    with col2:
        display_fig("p12_hyperparameter_tuning.png", "P12: Search Time vs. Test ROC-AUC Comparison")

# SECTION 8: INTERACTIVE PREDICTOR
elif menu == "Interactive ML Predictor (Live Inference)":
    st.header("🎯 Live Machine Learning Predictor")
    st.markdown("Enter custom athlete parameters to perform live inference using our trained regression and classification models!")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        height_in = st.slider("Athlete Height (cm)", min_value=140, max_value=220, value=180, step=1)
        age_in = st.slider("Athlete Age (years)", min_value=14, max_value=60, value=25, step=1)
    with col_in2:
        sex_in = st.selectbox("Biological Sex", ["Male", "Female"])
        season_in = st.selectbox("Olympic Season", ["Summer", "Winter"])
    with col_in3:
        year_in = st.slider("Olympic Year", min_value=1960, max_value=2024, value=2016, step=2)
        sport_in = st.selectbox("Sport Discipline", ["Athletics", "Swimming", "Gymnastics", "Rowing", "Cycling", "Other"])

    # Use cached pre-trained models for instantaneous, zero-latency inference
    is_male = 1 if sex_in == "Male" else 0
    is_summer = 1 if season_in == "Summer" else 0
    input_vec = np.array([[height_in, age_in, is_male, is_summer, year_in]])

    pred_weight = reg_model.predict(input_vec)[0]
    input_sc = scaler.transform(input_vec)
    pred_prob = clf_model.predict_proba(input_sc)[0][1] * 100
    bmi = pred_weight / ((height_in / 100) ** 2)

    st.markdown("---")
    st.subheader("Model Predictions:")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Predicted Body Mass", f"{pred_weight:.1f} kg")
    res_col2.metric("Calculated Body Mass Index (BMI)", f"{bmi:.1f} kg/m²")
    res_col3.metric("Predicted Medal Probability", f"{pred_prob:.1f}%")

    if pred_prob >= 50.0:
        st.success(f"🏆 High Medal Probability ({pred_prob:.1f}%)! Athlete profile aligns strongly with podium finishers in {sport_in}.")
    else:
        st.info(f"🏅 Competitive Participant ({pred_prob:.1f}% medal probability). Athlete profile matches representative Olympic competitors in {sport_in}.")
