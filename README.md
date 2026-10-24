# 🏅 Olympic Medal Data Analysis Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning project analyzing 120+ years of modern Olympic Games data (1896–2016). Covers missing data imputation, anti-leakage preprocessing, linear and multiple regression, supervised classification under class imbalance, unsupervised clustering, ensemble learning, cross-validation, and hyperparameter tuning.

---

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Aditipaul17/AditiPaul_olympic_medal_analysis.git
cd AditiPaul_olympic_medal_analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch interactive web dashboard (runs on http://localhost:8501)
streamlit run app.py

# Or run the complete CLI ML pipeline
python main.py
```

---

## 📁 Project Structure

```
├── app.py                          # Streamlit web dashboard with live ML predictor
├── main.py                         # Standalone end-to-end ML pipeline
├── olympic_medal_analysis.ipynb    # Executed master Jupyter Notebook (with inline charts)
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
├── data/
│   └── olympics.csv                # Historical Olympic Games dataset (271,116 rows)
└── outputs/
    └── figures/                    # 13 Generated evaluation plots (PNG)
```

---

## 📊 Machine Learning Topics & Results

| Topic | Algorithm / Method | Target / Feature | Key Metric | Key Finding |
|---|---|---|---|---|
| **Missing Data Imputation** | Mean, Median & Mode | Biometrics (`height`, `weight`, `age`) | Variance preservation | Median preserves IQR ($70.0\text{ kg}$) without artificial spikes. |
| **Preprocessing & Scaling** | `StandardScaler`, `MinMaxScaler` | `age`, `height`, `weight` | Train vs. Test scale | Train $\mu=0.00$, Test $\mu=-0.005$ (zero data leakage). |
| **Simple Linear Regression** | Ordinary Least Squares (OLS) | Height $\to$ Weight | $R^2 = 0.635$, $\text{RMSE} = 8.50\text{ kg}$ | Athlete mass increases $+1.08\text{ kg}$ per cm of height. |
| **Multiple Linear Regression** | Multivariate OLS | Height, Age, Sex, Sport $\to$ Weight | $R_{\text{adj}}^2 = 0.708$, $\text{RMSE} = 7.60\text{ kg}$ | Male sex adds $+7.61\text{ kg}$ controlling for height and sport. |
| **K-Nearest Neighbors** | KNN Classifier ($K=5$) | Medal Victory ($0/1$) | $\text{ROC-AUC} = 0.613$, F1 $= 0.218$ | Standardized features required; large $K$ dilutes minority class. |
| **Naive Bayes** | Gaussian Naive Bayes | Medal Victory ($0/1$) | $\text{ROC-AUC} = 0.612$, Fast fit | Fast $\mathcal{O}(Nd)$ baseline; robust despite height-weight correlation. |
| **Logistic Regression** | Cost-Balanced Logistic Reg | Medal Victory ($0/1$) | Recall: $0.1\% \to 58.2\%$, $\text{OR}_{\text{height}} = 1.15$ | Balancing corrects majority-class bias; height is a positive driver. |
| **K-Means Clustering** | K-Means ($K=3$) | Morphological Archetypes | Inertia Elbow, Silhouette | Discovers 3 biotypes: Power ($95\text{kg}$), Compact ($57\text{kg}$), Tall ($78\text{kg}$). |
| **Gaussian Mixture Models** | GMM (EM, Full Covariance) | Soft Morphology Clustering | BIC Minimum ($K=4$) | Anisotropic covariance captures natural height-weight elongation. |
| **Ensemble Classifiers** | Bagging & Random Forest | Medal Victory ($0/1$) | $\text{ROC-AUC}: 0.590 \to 0.662$ | Ensemble bagging + feature subsampling reduces variance by $>7\%$. |
| **Cross-Validation** | K-Fold vs. Stratified K-Fold | Validation Stability | Score std: $\pm 0.017 \to \pm 0.010$ | Stratification cuts variance by $40\%$ and locks class balance at $13.58\%$. |
| **Hyperparameter Tuning** | GridSearchCV vs. RandomizedSearch | Random Forest Tuning | Search Time: $13.8\text{s} \to 6.1\text{s}$ | Randomized search finds matching AUC ($0.6056$) in $56\%$ less time. |

---

## 🖥️ Interactive Web Dashboard (`app.py`)

Run `streamlit run app.py` to access the local web interface on **`http://localhost:8501`**:
- **Data Explorer:** Real-time participation and medal distribution charts.
- **Topic Walkthroughs:** Visual analytics and metrics across all ML techniques.
- **Live Predictor:** Interactive sliders for athlete height, age, sex, season, and sport to predict body mass and medal likelihood in real time.

---

## 📦 Dataset Information

- **Source:** [TidyTuesday Olympic Games Dataset](https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-07-27/olympics.csv)
- **Size:** 271,116 athlete records across 120 years (1896–2016).
- **Attributes:** 15 columns (demographics, physical measurements, event categories, medal achievements).

---

## 👤 Author
**Aditi Paul**  
- GitHub: [@Aditipaul17](https://github.com/Aditipaul17)  
- Repository: [AditiPaul_olympic_medal_analysis](https://github.com/Aditipaul17/AditiPaul_olympic_medal_analysis)
