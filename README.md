# Olympic Medal Data Analysis Using Machine Learning

An end-to-end Machine Learning research and practical project analyzing over 120 years of modern Olympic Games history (1896–2016). This repository implements **12 foundational and advanced machine learning practicals** using Scikit-Learn, Pandas, NumPy, Matplotlib, and Seaborn.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description & Schema](#dataset-description--schema)
- [Repository Structure](#repository-structure)
- [Installation & Quick Start](#installation--quick-start)
- [Comprehensive Practical Breakdown](#comprehensive-practical-breakdown)
  - [Practical 1: Missing Value Detection & Statistical Imputation](#practical-1-missing-value-detection--statistical-imputation)
  - [Practical 2: Data Preprocessing, Feature Scaling & Anti-Leakage](#practical-2-data-preprocessing-feature-scaling--anti-leakage)
  - [Practical 3: Simple Linear Regression](#practical-3-simple-linear-regression)
  - [Practical 4: Multiple Linear Regression](#practical-4-multiple-linear-regression)
  - [Practical 5: K-Nearest Neighbors (KNN) Classifier](#practical-5-k-nearest-neighbors-knn-classifier)
  - [Practical 6: Gaussian Naive Bayes Classification](#practical-6-gaussian-naive-bayes-classification)
  - [Practical 7: Regularized Logistic Regression & Class Imbalance](#practical-7-regularized-logistic-regression--class-imbalance)
  - [Practical 8: K-Means Clustering & Morphological Profiling](#practical-8-k-means-clustering--morphological-profiling)
  - [Practical 9: Gaussian Mixture Models (GMM) Soft Clustering](#practical-9-gaussian-mixture-models-gmm-soft-clustering)
  - [Practical 10: Bagging & Random Forest Classifiers](#practical-10-bagging--random-forest-classifiers)
  - [Practical 11: K-Fold & Stratified K-Fold Cross-Validation](#practical-11-k-fold--stratified-k-fold-cross-validation)
  - [Practical 12: Hyperparameter Tuning (GridSearchCV vs. RandomizedSearchCV)](#practical-12-hyperparameter-tuning-gridsearchcv-vs-randomizedsearchcv)
- [Empirical Results Summary Matrix](#empirical-results-summary-matrix)
- [Visual Artifacts](#visual-artifacts)
- [Verification & Integrity Principles](#verification--integrity-principles)

---

## 🔬 Project Overview
This project applies statistical and machine learning methodologies to the historical record of the Olympic Games. It addresses both supervised tasks (predicting athlete body mass through biomechanical allometry, and classifying medalists under severe class imbalance) and unsupervised tasks (discovering athletic biotypes and morphological clustering).

Key technical objectives:
1. **Prevent Data Leakage:** Ensure transformation estimators (scalers, encoders, and imputers) learn parameters strictly on training partitions before transforming validation/test data.
2. **Handle Real-World Class Imbalance:** Olympic medalists represent only ~14.7% of athlete participations. We demonstrate how cost-sensitive learning (`class_weight='balanced'`) prevents model collapse into the majority class.
3. **Rigorous Empirical Verification:** Zero fabricated results. Every metric, table, coefficient, and plot is computed directly from the dataset.

---

## 📊 Dataset Description & Schema
- **Source:** [TidyTuesday Olympic Games Dataset (rfordatascience/tidytuesday)](https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-07-27/olympics.csv)
- **Observations:** 271,116 rows
- **Attributes:** 15 columns

| Column | Type | Description | Missing Count (%) |
|---|---|---|---|
| `id` | Integer | Unique athlete identifier | 0 (0.00%) |
| `name` | String | Athlete full name | 0 (0.00%) |
| `sex` | String | Biological sex (`M` / `F`) | 0 (0.00%) |
| `age` | Float | Athlete age in years | 9,474 (3.49%) |
| `height` | Float | Athlete height in centimeters | 60,171 (22.19%) |
| `weight` | Float | Athlete body mass in kilograms | 62,875 (23.19%) |
| `team` | String | Country / Team represented | 0 (0.00%) |
| `noc` | String | National Olympic Committee 3-letter code | 0 (0.00%) |
| `games` | String | Games edition name (e.g., `1992 Summer`) | 0 (0.00%) |
| `year` | Integer | Olympic Games year (1896 – 2016) | 0 (0.00%) |
| `season` | String | Season (`Summer` / `Winter`) | 0 (0.00%) |
| `city` | String | Host city | 0 (0.00%) |
| `sport` | String | Olympic sporting discipline (66 sports) | 0 (0.00%) |
| `event` | String | Specific athletic event | 0 (0.00%) |
| `medal` | String | Medal achieved (`Gold`, `Silver`, `Bronze`, `NaN`) | 231,333 (85.33%) |

---

## 📁 Repository Structure
```
olympic-medal-ml-analysis/
├── data/
│   └── olympics.csv                      # Cached Olympic dataset (1896-2016)
├── outputs/
│   └── figures/                          # 13 Publication-ready visualization figures
│       ├── p1_missing_values_bar.png
│       ├── p1_imputation_distribution_comparison.png
│       ├── p2_scaling_comparison_boxplots.png
│       ├── p3_simple_linear_regression.png
│       ├── p4_multiple_linear_regression.png
│       ├── p5_knn_evaluation.png
│       ├── p6_naive_bayes_evaluation.png
│       ├── p7_logistic_regression.png
│       ├── p8_kmeans_clustering.png
│       ├── p9_gmm_clustering.png
│       ├── p10_ensemble_comparison.png
│       ├── p11_cross_validation_comparison.png
│       └── p12_hyperparameter_tuning.png
├── main.py                               # Complete runnable Python script executing all 12 practicals
├── olympic_medal_analysis.ipynb          # Master interactive Jupyter Notebook (41 cells)
├── requirements.txt                      # Project dependency specification
└── README.md                             # Comprehensive technical documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone or Download the Repository
```bash
git clone <repo-url>
cd olympic-medal-ml-analysis
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the Complete Pipeline
```bash
# Run the complete end-to-end Python pipeline:
python main.py

# Launch the interactive Jupyter Notebook:
jupyter notebook olympic_medal_analysis.ipynb
```

---

## 📖 Comprehensive Practical Breakdown

### Practical 1: Missing Value Detection & Statistical Imputation
- **Aim:** Identify missing data patterns across biometric attributes (`age`, `height`, `weight`) and structural targets (`medal`), and evaluate the statistical effects of Mean, Median, and Mode imputation.
- **Theory:** Missing data can be MCAR, MAR, or MNAR. For historical Olympic data, pre-1960 games suffered from systematic omissions. In the `medal` column, missingness is structural ($NaN = \text{Non-medalist}$, 85.33% of records).
- **Mathematical Imputation Formulas:**
  - *Mean Imputation:* $\hat{x}_i = \bar{x} = \frac{1}{n} \sum x_i$ (preserves the mean, compresses variance $\sigma^2$).
  - *Median Imputation:* $\hat{x}_i = \text{Median}(X)$ (minimizes $L_1$ absolute error; outlier-resistant).
  - *Mode Imputation:* $\hat{x}_i = \arg\max_c \text{freq}(c)$ (categorical / discrete features).
- **Empirical Findings:**
  - Mean imputation on athlete weight compresses the standard deviation from $14.35\text{ kg}$ down to $12.57\text{ kg}$ and introduces an artificial density spike.
  - Median imputation cleanly preserves percentiles at $70.00\text{ kg}$ without introducing boundary distortion.

### Practical 2: Data Preprocessing, Feature Scaling & Anti-Leakage
- **Aim:** Encode categorical attributes into numerical vectors, standardize and normalize continuous variables, and implement strict anti-leakage pipelines.
- **Theory:** Nominal variables (`sport`, `season`, `sex`) are transformed via One-Hot Encoding ($k-1$ dummies) and binary mapping ($M \to 1, F \to 0$). Continuous variables are scaled using:
  - *StandardScaler:* $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$
  - *MinMaxScaler:* $x_{\text{scaled}} = \frac{x - x_{\min,\text{train}}}{x_{\max,\text{train}} - x_{\min,\text{train}}}$
- **Anti-Leakage Protocol:** Scalers are fitted strictly on `X_train`. Transformed training data yields $\mu=0.000, \sigma=1.000$, while test data yields $\mu=-0.005, \sigma=1.003$, verifying zero information leakage.

### Practical 3: Simple Linear Regression
- **Aim:** Predict athlete Weight (kg) as an allometric biomechanical function of Height (cm).
- **Mathematical Formulation:**
  $$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$
  $$\beta_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} = 1.0772\text{ kg/cm}, \quad \beta_0 = -118.2326\text{ kg}$$
- **Metrics:** $R^2 = 0.6348$, $\text{MAE} = 5.868\text{ kg}$, $\text{RMSE} = 8.501\text{ kg}$.
- **Conclusion:** Elite Olympic athletes gain approximately $1.08\text{ kg}$ of body mass per additional centimeter of stature. Residuals exhibit slight heteroscedasticity at high masses due to powerlifters and heavyweight throwers.

### Practical 4: Multiple Linear Regression
- **Aim:** Predict athlete Weight incorporating multiple covariates (`height`, `age`, `sex_male`, `season_summer`, `year`, and sport categories).
- **Mathematical Formulation:**
  $$\hat{\beta} = (X^T X)^{-1} X^T Y, \quad R_{\text{adj}}^2 = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - p - 1}\right]$$
- **Metrics:** $R_{\text{adj}}^2 = 0.7076$ (up from $0.6348$), $\text{MAE} = 5.250\text{ kg}$, $\text{RMSE} = 7.603\text{ kg}$.
- **Conclusion:** Biological sex adds $+7.61\text{ kg}$ of expected mass controlling for height and discipline. The residual error distribution displays near-Gaussian symmetry around zero.

### Practical 5: K-Nearest Neighbors (KNN) Classifier
- **Aim:** Classify whether an athlete wins a medal based on biometric and competition features; optimize neighborhood size $K$.
- **Mathematical Formulation:**
  $$d(x, x') = \sqrt{\sum (x_j - x'_j)^2}, \quad P(Y = c \mid X = x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}(y_i = c)$$
- **Metrics Across $K$:**
  - $K=3$: Accuracy: $0.8354$, F1: $0.2185$, ROC-AUC: $0.5898$
  - $K=5$: Accuracy: $0.8499$, F1: $0.2183$, ROC-AUC: $0.6133$ (Optimal balance)
  - $K=17$: Accuracy: $0.8624$, F1: $0.0906$, ROC-AUC: $0.6231$
- **Conclusion:** Distance standardization is mandatory. Larger $K$ smooths probabilities but suppresses minority medal detection due to class imbalance.

### Practical 6: Gaussian Naive Bayes Classification
- **Aim:** Implement a probabilistic Gaussian Naive Bayes classifier on Olympic athlete features, measuring class priors and posterior predictions.
- **Mathematical Formulation:**
  $$P(Y = c \mid X) \propto P(Y = c) \prod_{j=1}^d \frac{1}{\sqrt{2\pi\sigma_{cj}^2}} \exp\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$
- **Metrics:** Accuracy: $0.8523$, Recall: $0.0543$, F1: $0.0910$, ROC-AUC: $0.6115$.
- **Conclusion:** Features train in milliseconds with $\mathcal{O}(Nd)$ complexity. Moderate covariance between height and weight ($r \approx 0.79$) partially violates conditional independence but preserves competitive ranking capability.

### Practical 7: Regularized Logistic Regression & Class Imbalance
- **Aim:** Overcome the ~14:86 medalist class imbalance using cost-sensitive learning (`class_weight='balanced'`) and evaluate feature odds ratios $\exp(\beta_j)$.
- **Mathematical Formulation:**
  $$p(x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}, \quad \text{OR}_j = \exp(\beta_j)$$
- **Impact of Balancing:**
  - *Unweighted Model:* Accuracy: $86.41\%$, Recall: $0.0011$ (collapses to majority class rejection).
  - *Balanced Model:* Accuracy: $58.26\%$, Recall: $58.19\%$, F1: $0.2755$, ROC-AUC: $0.6111$.
- **Odds Ratios:** Height ($\text{OR} = 1.15$) and weight ($\text{OR} = 1.07$) yield odds ratios $>1.0$, confirming positive physiological advantages.

### Practical 8: K-Means Clustering & Morphological Profiling
- **Aim:** Unsupervised discovery of athlete morphological archetypes via K-Means; optimize $K$ using Inertia (Elbow Method) and Silhouette Analysis.
- **Mathematical Formulation:**
  $$\min \sum_{k=1}^K \sum_{x \in S_k} \|x - \mu_k\|^2, \quad s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- **Discovered Archetypes ($K=3$):**
  1. *Heavyweight / Power Biotype:* Mean Height: $181.2\text{ cm}$, Weight: $95.4\text{ kg}$, $\text{BMI}: 28.8$ (Weightlifters, super-heavyweights).
  2. *Lightweight / Compact Biotype:* Mean Height: $167.3\text{ cm}$, Weight: $57.1\text{ kg}$, $\text{BMI}: 20.6$ (Gymnasts, distance runners).
  3. *Tall / Athletic Biotype:* Mean Height: $185.1\text{ cm}$, Weight: $78.3\text{ kg}$, $\text{BMI}: 22.9$ (Rowers, swimmers, basketballers).

### Practical 9: Gaussian Mixture Models (GMM) Soft Clustering
- **Aim:** Fit probabilistic mixture distributions using Expectation-Maximization (EM) and evaluate model complexity using AIC and BIC.
- **Mathematical Formulation:**
  $$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k), \quad \text{BIC} = p\ln(N) - 2\ln\hat{L}$$
- **Findings:** BIC identifies $K=4$ full-covariance components as optimal. GMM captures elliptical anisotropic boundaries reflecting real-world covariance between height and weight.

### Practical 10: Bagging & Random Forest Classifiers
- **Aim:** Construct ensemble architectures to minimize model variance, measure Out-of-Bag (OOB) generalization, and extract Gini feature importances.
- **Theoretical Benchmark:**
  - *Single Decision Tree:* Accuracy: $0.8353$, ROC-AUC: $0.5902$
  - *Bagging Classifier:* Accuracy: $0.8656$, ROC-AUC: $0.6397$, OOB: $0.8631$
  - *Random Forest (Balanced):* Accuracy: $0.6272$, Recall: $0.5925$, ROC-AUC: $0.6621$, OOB: $0.6552$
- **Top Feature Drivers:** Athlete Age ($32.2\%$), Height ($24.6\%$), Body Weight ($23.8\%$), Competition Year ($14.1\%$).

### Practical 11: K-Fold & Stratified K-Fold Cross-Validation
- **Aim:** Empirically quantify metric variance and class proportion preservation between standard K-Fold and Stratified K-Fold.
- **Experimental Results (5 Folds):**
  - *Standard K-Fold:* Mean ROC-AUC = $0.6025 \pm 0.0171$ (Positive class range: $13.30\% - 13.97\%$).
  - *Stratified K-Fold:* Mean ROC-AUC = $0.6025 \pm 0.0103$ (Positive class locked at $13.58\%$).
- **Conclusion:** Stratification reduces fold-to-fold standard deviation by $\sim 40\%$ and eliminates artificial distribution shift.

### Practical 12: Hyperparameter Tuning (GridSearchCV vs. RandomizedSearchCV)
- **Aim:** Tune Random Forest hyperparameters across `n_estimators`, `max_depth`, `min_samples_split`, and `criterion`.
- **Benchmark Results:**
  - *GridSearchCV:* 24 parameter combinations (72 fits) $\to$ Search Time: $13.75\text{s}$, Test ROC-AUC: $0.6054$.
  - *RandomizedSearchCV:* 10 random combinations (30 fits) $\to$ Search Time: $6.08\text{s}$, Test ROC-AUC: $0.6056$.
- **Conclusion:** RandomizedSearchCV matches exhaustive grid search accuracy in less than half the execution time ($55.8\%$ reduction in compute budget).

---

## 📈 Empirical Results Summary Matrix

| Practical # | Machine Learning Paradigm | Model Architecture | Primary Task | Key Performance Metric(s) | Primary Conclusion |
|---|---|---|---|---|---|
| **P1** | Data Cleaning | Statistical Imputation | Missing Biometrics | $\sigma_{\text{orig}}=14.35 \to \sigma_{\text{mean}}=12.57$ | Median preserves IQR without artificial variance collapse. |
| **P2** | Preprocessing | Standard / MinMax Scaling | Feature Normalization | Train $\mu=0.000$, Test $\mu=-0.005$ | Zero data leakage achieved across validation splits. |
| **P3** | Univariate Regression | Ordinary Least Squares (OLS) | Height $\to$ Weight | $R^2 = 0.6348$, $\text{RMSE} = 8.501\text{ kg}$ | Mass increases $+1.08\text{ kg}$ per cm of athlete height. |
| **P4** | Multivariate Regression | Multiple Linear Regression | Biometrics $\to$ Weight | $R_{\text{adj}}^2 = 0.7076$, $\text{RMSE} = 7.603\text{ kg}$ | Male biological sex adds $+7.61\text{ kg}$ controlling for stature. |
| **P5** | Instance Classification | K-Nearest Neighbors | Medalist Prediction | $\text{ROC-AUC} = 0.6133$, Best $K=5$ | Scaled distances required; large $K$ dilutes imbalanced minority. |
| **P6** | Bayesian Modeling | Gaussian Naive Bayes | Medalist Prediction | $\text{ROC-AUC} = 0.6115$, Time $< 0.1\text{s}$ | Fast baseline; resilient despite biometric covariance. |
| **P7** | Generalized Linear | Balanced Logistic Regression | Medalist Prediction | Recall $0.1\% \to 58.2\%$, $\text{OR}_{\text{height}}=1.15$ | Balancing prevents majority-class collapse; height is positive driver. |
| **P8** | Clustering | K-Means ($K=3$) | Morphology Discovery | Inertia Elbow, Silhouette Score | Discovers Power, Compact, and Tall athletic biotypes. |
| **P9** | Probabilistic Clustering | Gaussian Mixture Models | Soft Clustering | BIC Minimum ($K=4$), Ellipsoidal | Anisotropic covariance models real-world height-weight correlation. |
| **P10** | Ensemble Classifiers | Bagging & Random Forest | Medalist Prediction | $\text{ROC-AUC}: 0.590 \to 0.662$ | Ensemble bagging + feature subspace sampling cuts variance. |
| **P11** | Model Validation | Stratified 5-Fold CV | Validation Stability | Score Std: $\pm 0.017 \to \pm 0.010$ | Stratification stabilizes class proportions across folds. |
| **P12** | Hyperparameter Search | Random vs. Grid Search | RF Tuning | Search Time: $13.8\text{s} \to 6.1\text{s}$ | Random search achieves matching AUC with 56% less runtime. |

---

## 🖼 Visual Artifacts
All 13 high-resolution plots generated by `main.py` are saved in `outputs/figures/`:
1. `p1_missing_values_bar.png` - Percentage missingness per attribute.
2. `p1_imputation_distribution_comparison.png` - Kernel density estimation comparison of imputation methods.
3. `p2_scaling_comparison_boxplots.png` - Raw vs. StandardScaler vs. MinMaxScaler boxplots.
4. `p3_simple_linear_regression.png` - OLS regression line and residual scatter plot.
5. `p4_multiple_linear_regression.png` - Predicted vs. actual body mass and residual histogram.
6. `p5_knn_evaluation.png` - Trajectory of Accuracy, F1, and ROC-AUC across $K$, plus confusion matrix.
7. `p6_naive_bayes_evaluation.png` - GaussianNB ROC curve and confusion matrix.
8. `p7_logistic_regression.png` - Odds ratios bar chart and unweighted vs. balanced ROC curves.
9. `p8_kmeans_clustering.png` - Elbow inertia and silhouette curves with 2D cluster centroid scatter.
10. `p9_gmm_clustering.png` - AIC/BIC information criteria and soft ellipsoidal component distributions.
11. `p10_ensemble_comparison.png` - Decision Tree vs. Bagging vs. Random Forest ROC curves and Gini importances.
12. `p11_cross_validation_comparison.png` - Cross-validation score boxplots and fold positive ratios.
13. `p12_hyperparameter_tuning.png` - Execution runtime and ROC-AUC comparisons for Grid vs. Random search.

---

## 🛡 Verification & Integrity Principles
- **No Fabricated Data:** All models train and evaluate on real historical records from the 1896–2016 Olympic Games dataset.
- **Deterministic Reproducibility:** Random seeds (`random_state=42`) are applied across all train/test splits, cross-validation folds, and stochastic estimators.
- **Platform Agnostic:** Standard output encoding is UTF-8 safeguarded for seamless execution on Windows (PowerShell/CMD), macOS, and Linux terminals.
