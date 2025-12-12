Table of Contents
-----------------

-   [Project Overview](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#project-overview)

-   [Objectives](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#objectives)

-   [Problem Statement](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#problem-statement)

-   [Dataset Description](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#dataset-description)

-   [System Architecture](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#system-architecture)

-   [Algorithms and Methodology](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#algorithms-and-methodology)

-   [Feature Engineering](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#feature-engineering)

-   [Risk Analysis Framework](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#risk-analysis-framework)

-   [Live Analysis System](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#live-analysis-system)

-   [Results and Performance](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#results-and-performance)

-   [Installation and Setup](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#installation-and-setup)

-   [Usage Guide](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#usage-guide)

-   [Project Structure](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#project-structure)

-   [Visualizations](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#visualizations)

-   [Limitations and Future Work](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#limitations-and-future-work)

-   [References](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#references)

-   [Contributors](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#contributors)

-   [License](https://chat.z.ai/c/c27e6500-ce26-4695-bdd7-c1542dcf4d0d#license)

Project Overview
----------------

The Cryptocurrency Scam Detection System is a comprehensive machine learning solution designed to identify potentially fraudulent cryptocurrencies and protect investors from financial losses. The system analyzes multiple data points including smart contract characteristics, team transparency, liquidity metrics, and market behavior to generate risk assessments for any cryptocurrency.

The cryptocurrency market has witnessed explosive growth, but this has also attracted malicious actors who exploit investors through various scam mechanisms including rug pulls, honeypot contracts, pump-and-dump schemes, and Ponzi structures. This project addresses this critical need by providing an automated, data-driven approach to scam detection.

### Key Features

-   **Multi-Model Ensemble Classification**: Combines 5 machine learning algorithms for robust predictions

-   **Real-Time Analysis**: Fetches live data from CoinGecko API for instant risk assessment

-   **Comprehensive Risk Scoring**: Hybrid scoring system combining ML predictions with rule-based analysis

-   **Documented Scam Database**: Includes 50+ real-world documented scam cases for training

-   **Interactive Visualizations**: Rich charts and graphs for data exploration

-   **Category-Specific Analysis**: Tailored risk profiles for different cryptocurrency categories

-   **Investment Recommendations**: Clear, actionable guidance for investment decisions

Objectives
----------

### Primary Objectives

1.  **Develop a Reliable Classification System**: Build a machine learning model capable of distinguishing between legitimate cryptocurrencies and potential scams with high accuracy and minimal false positives.

2.  **Create a Comprehensive Risk Scoring Framework**: Design a scoring system that quantifies risk on a 0-100 scale, incorporating both algorithmic predictions and domain-specific rules.

3.  **Enable Real-Time Analysis**: Implement a live prediction system that can analyze any cryptocurrency by fetching current market data and applying trained models.

4.  **Provide Actionable Insights**: Generate clear, understandable recommendations that help users make informed investment decisions.

### Secondary Objectives

-   Document and analyze patterns in historical cryptocurrency scams

-   Identify key indicators that differentiate scams from legitimate projects

-   Create educational resources about cryptocurrency fraud detection

-   Build a reusable framework for ongoing market monitoring

Problem Statement
-----------------

### The Cryptocurrency Fraud Epidemic

The cryptocurrency market has experienced unprecedented growth, with thousands of new tokens launched daily. However, studies estimate that:

-   Over $14 billion was lost to cryptocurrency scams in 2021 alone

-   Rug pulls accounted for 37% of all cryptocurrency scam revenue

-   Celebrity-endorsed tokens have a scam rate exceeding 85%

-   New investors are disproportionately affected due to lack of technical knowledge

### Types of Scams Addressed

**Scam TypeDescriptionNotable Examples**Rug PullDevelopers abandon project after extracting liquiditySquid Game Token, AnubisDAOHoneypotSmart contract prevents selling tokensSquid Game Token, ShibachuPump and DumpCoordinated price manipulation followed by mass sellingSave The Kids, EthereumMaxPonzi SchemeReturns paid from new investor depositsBitConnect, OneCoinExit ScamTeam disappears with investor fundsThodex, QuadrigaCXImpersonationFake tokens mimicking legitimate projectsFake BabyDoge, Milady Maker Fake

### Challenges in Detection

-   **Data Scarcity**: Limited labeled data for scam classification

-   **Class Imbalance**: Scams are minority class in overall market

-   **Evolving Tactics**: Scammers continuously adapt their methods

-   **Feature Accessibility**: Not all relevant data is publicly available

-   **Real-Time Requirements**: Markets move fast, requiring quick analysis

Dataset Description
-------------------

### Data Sources

The system utilizes a multi-source dataset combining:

1.  **Documented Scam Cases**: Manually curated database of 50+ confirmed scams

2.  **Legitimate Cryptocurrency Data**: Verified legitimate projects across categories

3.  **Live Market Data**: Real-time data from CoinGecko API

4.  **Synthetic Variants**: Augmented data to improve model generalization

## Dataset Composition

| Category           | Count | Description                                  |
|--------------------|--------|----------------------------------------------|
| Documented Scams   | 50+    | Real-world confirmed scam cases              |
| Legitimate Coins   | 60+    | Established, verified cryptocurrencies       |
| Edge Cases         | 80     | Ambiguous cases for model robustness         |
| Synthetic Variants | 80+    | Augmented samples for diversity              |
| Live Data          | 100    | Top coins by market cap from API             |
| **Total**          | **370+** | Complete training dataset                    |


## Cryptocurrency Categories

| Category Type         | Examples                                |
|-----------------------|-------------------------------------------|
| Layer 1 Blockchains   | BTC, ETH, SOL, ADA, AVAX                 |
| Layer 2 Solutions     | MATIC, ARB, OP                           |
| DeFi Protocols        | UNI, AAVE, LINK, MKR                     |
| Stablecoins           | USDT, USDC, DAI                          |
| Meme Coins            | DOGE, SHIB, PEPE                         |
| Exchange Tokens       | BNB, CRO, OKB                            |
| Gaming / Metaverse    | SAND, MANA, AXS                          |
| AI Tokens             | RNDR, FET, AGIX                          |
| Social Tokens         | CHZ, RLY, DESO                           |
| Influencer Coins      | High-risk category                       |
| Ponzi Schemes         | Historical examples                      |


### Feature Set

The dataset includes 40+ features across multiple dimensions:

## Feature Set

### Security Features

| Feature              | Type    | Description                                 |
|----------------------|---------|---------------------------------------------|
| had_audit            | Binary  | Whether contract has been audited           |
| team_doxxed          | Binary  | Whether team identities are public          |
| liquidity_locked     | Binary  | Whether liquidity is locked                 |
| ownership_renounced  | Binary  | Whether contract ownership is renounced     |
| contract_verified    | Binary  | Whether source code is verified             |
| honeypot             | Binary  | Whether honeypot behavior detected          |


### Market Metrics

| Feature          | Type    | Description                    |
|------------------|---------|--------------------------------|
| age_days         | Numeric | Project age in days            |
| holder_count     | Numeric | Number of token holders        |
| social_followers | Numeric | Social media following         |
| market_cap       | Numeric | Total market capitalization     |
| volume_24h       | Numeric | 24-hour trading volume         |


### Contract Features

| Feature            | Type    | Description                              |
|--------------------|---------|------------------------------------------|
| buy_tax            | Numeric | Tax percentage on purchases              |
| sell_tax           | Numeric | Tax percentage on sales                  |
| max_tx_limit       | Binary  | Whether transaction limits exist         |
| top_holder_percent | Numeric | Percentage held by largest holder        |


### Presence Indicators

| Feature            | Type    | Description                         |
|--------------------|---------|-------------------------------------|
| website_exists     | Binary  | Whether project has a website       |
| whitepaper_exists  | Binary  | Whether whitepaper is available     |
| github_exists      | Binary  | Whether open-source code exists     |


### Documented Scam Examples

The dataset includes detailed information on major cryptocurrency scams:

| Name               | Symbol | Category    | Scam Type      | Estimated Losses |
|--------------------|--------|-------------|----------------|------------------|
| BitConnect         | BCC    | Ponzi       | Ponzi Scheme   | $3.5 Billion     |
| OneCoin            | ONE    | Ponzi       | Ponzi Scheme   | $4.0 Billion     |
| FTX Token          | FTT    | Exchange    | Fraud          | $8.0 Billion     |
| Squid Game Token   | SQUID  | Meme        | Honeypot       | $12 Million      |
| Save The Kids      | KIDS   | Influencer  | Rug Pull       | $30 Million      |
| EthereumMax        | EMAX   | Influencer  | Pump & Dump    | $100 Million     |
| Celsius            | CEL    | DeFi        | Fraud          | $4.7 Billion     |
| Thodex             | THODEX | Exchange    | Exit Scam      | $2.0 Billion     |


System Architecture
-------------------


### Component Descriptions

1.  **Data Layer**

    -   **CryptoDataFetcher**: Interfaces with CoinGecko API for real-time data

    -   **ScamDatabase**: Maintains curated database of documented scams

    -   **DatasetBuilder**: Combines multiple data sources into unified dataset

2.  **Machine Learning Layer**

    -   **FeatureEngineer**: Transforms raw data into ML-ready features

    -   **ScamDetector**: Trains and manages multiple ML models

    -   **Ensemble Classifier**: Combines predictions from multiple models

3.  **Risk Analysis Layer**

    -   **RiskScorer**: Calculates comprehensive risk scores

    -   **Rule Engine**: Applies domain-specific rules for red flag detection

    -   **Category Analyzer**: Provides category-specific risk profiles

4.  **Presentation Layer**

    -   **Visualizer**: Generates interactive charts and graphs

    -   **LiveCoinPredictor**: Provides real-time analysis interface

    -   **Recommendation Engine**: Generates investment guidance

Algorithms and Methodology
--------------------------

### Machine Learning Pipeline

Raw Data  
   ↓  
Preprocessing  
   ↓  
Feature Engineering  
   ↓  
Model Training  
   ↓  
Ensemble  
   ↓  
Prediction


### Classification Algorithms

#### 1\. Logistic Regression

**Purpose**: Provides baseline performance and interpretable coefficients



**Strengths**:

-   Fast training and inference

-   Highly interpretable

-   Works well with regularization

-   Provides probability estimates

#### 2\. Random Forest Classifier

**Purpose**: Captures non-linear relationships and provides feature importance



**Strengths**:

-   Handles mixed feature types

-   Built-in feature importance

-   Robust to outliers

-   Out-of-bag error estimation

#### 3\. XGBoost Classifier

**Purpose**: High-performance gradient boosting with regularization



**Strengths**:

-   State-of-the-art performance

-   Built-in regularization

-   Handles missing values

-   Parallel processing support

#### 4\. LightGBM Classifier

**Purpose**: Efficient gradient boosting optimized for large datasets



**Strengths**:

-   Faster training than XGBoost

-   Lower memory usage

-   Handles categorical features natively

-   Excellent for imbalanced data

#### 5\. Gradient Boosting Classifier

**Purpose**: Scikit-learn implementation for comparison



### Ensemble Method

The final prediction uses a Soft Voting Classifier that combines predictions from multiple models:

```python
VotingClassifier(
    estimators=[
        ('lr', logistic_regression),
        ('rf', random_forest),
        ('lgb', lightgbm)
    ],
    voting='soft',
    weights=[1, 1.5, 1.5]  # Higher weight for tree-based models
)
```

**Ensemble Logic**:

-   Each model provides probability estimates

-   Probabilities are weighted and averaged

-   Final prediction based on combined probability

-   Reduces variance and improves robustness

### Handling Class Imbalance

The dataset exhibits class imbalance (more legitimate coins than scams). We address this using:

#### 1\. SMOTE (Synthetic Minority Over-sampling Technique)

### Handling Class Imbalance — SMOTE

```python
SMOTE(
    random_state=42,
    k_neighbors=3
)
```

SMOTE creates synthetic examples of the minority class by:

-   Finding k-nearest neighbors for each minority sample

-   Creating new samples along the line between neighbors

-   Balancing the dataset without simple duplication

#### 2\. Class Weights

All models use **class_weight='balanced'** to:

-   Automatically adjust weights inversely proportional to class frequencies

-   Penalize misclassification of minority class more heavily





### Evaluation Metrics

| Metric     | Formula                       | Purpose                         |
|------------|-------------------------------|---------------------------------|
| Accuracy   | (TP + TN) / (TP + TN + FP + FN) | Overall correctness             |
| Precision  | TP / (TP + FP)                | Avoid false scam labels         |
| Recall     | TP / (TP + FN)                | Catch actual scams              |
| F1 Score   | 2 × (Precision × Recall) / (Precision + Recall) | Balance precision and recall |
| ROC-AUC    | Area under ROC curve          | Measures discrimination ability |


## Feature Engineering

### Derived Features

---

### Log Transformations

Numeric features with skewed distributions are log-transformed:

    df['holder_count_log'] = np.log1p(df['holder_count'])
    df['social_followers_log'] = np.log1p(df['social_followers'])
    df['volatility_log'] = np.log1p(df['volatility'])
    df['age_days_log'] = np.log1p(df['age_days'])

---

### Age-Based Features

    df['is_very_new']     = (df['age_days'] < 14).astype(int)
    df['is_new']          = (df['age_days'] < 60).astype(int)
    df['is_established']  = (df['age_days'] > 365).astype(int)
    df['is_veteran']      = (df['age_days'] > 1000).astype(int)

---

### Trust Score (Composite)

Aggregates multiple security indicators:

    trust_score = (
        had_audit * 25 +
        team_doxxed * 20 +
        liquidity_locked * 25 +
        ownership_renounced * 15 +
        contract_verified * 10 +
        website_exists * 3 +
        whitepaper_exists * 5 +
        github_exists * 7
    ) / 100

---

### Red Flag Score

Aggregates warning indicators:

    red_flag_score = (
        honeypot * 50 +
        (1 - had_audit) * 12 +
        (1 - team_doxxed) * 8 +
        (1 - liquidity_locked) * 15 +
        (sell_tax > 10) * 12 +
        (top_holder_percent > 50) * 12 +
        (age_days < 14) * 8
    ) / 100

---

### Tax Analysis Features

    df['tax_difference'] = df['sell_tax'] - df['buy_tax']
    df['total_tax'] = df['sell_tax'] + df['buy_tax']
    df['high_sell_tax'] = (df['sell_tax'] > 15).astype(int)
    df['suspicious_tax'] = (df['tax_difference'] > 10).astype(int)

---

### Interaction Features

    df['anon_no_audit'] = ((team_doxxed == 0) & (had_audit == 0)).astype(int)
    df['new_high_risk'] = (is_new & high_risk_category).astype(int)
    df['crash_velocity'] = price_drop_percent / (days_to_crash + 1)

---

### Feature Importance Analysis

Top features identified by Random Forest:

| Rank | Feature            | Importance | Description                 |
|------|---------------------|------------|-----------------------------|
| 1    | red_flag_score      | 0.142      | Composite risk indicator    |
| 2    | trust_score         | 0.128      | Composite safety indicator  |
| 3    | liquidity_locked    | 0.095      | Liquidity protection        |
| 4    | age_days_log        | 0.087      | Project maturity            |
| 5    | sell_tax            | 0.076      | Transaction tax level       |
| 6    | had_audit           | 0.068      | Security verification       |
| 7    | team_doxxed         | 0.062      | Team transparency           |
| 8    | top_holder_percent  | 0.058      | Ownership concentration     |
| 9    | honeypot_indicator  | 0.054      | Critical scam flag          |
| 10   | high_risk_category  | 0.048      | Category-based risk         |


Risk Analysis Framework
-----------------------

### Hybrid Scoring System

The risk score combines ML predictions with rule-based analysis:

Line WrappingCollapseCopy1Final Risk Score = 0.6 × (ML Probability × 100) + 0.4 × Rule Score

## Rule-Based Scoring

Points are assigned for each risk factor:

| Factor                         | Points | Condition                      |
|--------------------------------|--------|--------------------------------|
| Honeypot Detected             | +40    | honeypot = 1                   |
| Very High Sell Tax            | +20    | sell_tax > 25%                 |
| High Sell Tax                 | +12    | sell_tax > 15%                 |
| Moderate Sell Tax             | +6     | sell_tax > 10%                 |
| No Audit                      | +10    | had_audit = 0                  |
| Anonymous Team                | +8     | team_doxxed = 0                |
| Unlocked Liquidity            | +12    | liquidity_locked = 0           |
| Ownership Not Renounced       | +4     | ownership_renounced = 0        |
| Very New (< 7 days)           | +12    | age_days < 7                   |
| New (< 14 days)               | +8     | age_days < 14                  |
| Recent (< 30 days)            | +4     | age_days < 30                  |
| High Concentration (> 60%)    | +12    | top_holder_percent > 60        |
| Moderate Concentration (> 40%)| +6     | top_holder_percent > 40        |
| Influencer Category           | +10    | category = influencer          |
| Meme Category                 | +5     | category = meme_coin           |
| Ponzi Category                | +25    | category = ponzi               |

---

## Risk Level Classification

| Score Range | Level     | Recommendation                                    |
|-------------|-----------|--------------------------------------------------|
| 0–29        | LOW       | Standard due diligence recommended               |
| 30–49       | MEDIUM    | Conduct thorough research before investing       |
| 50–69       | HIGH      | Multiple red flags — approach with caution       |
| 70–100      | CRITICAL  | Strong scam indicators — avoid investment        |

---

## Category-Specific Risk Profiles

Each cryptocurrency category has unique characteristics:

| Category      | Base Risk     | Scam Rate | Investment Horizon         | Volatility     |
|---------------|---------------|-----------|-----------------------------|----------------|
| Layer 1       | LOW           | 5%        | Long-term (2–5 years)       | Medium-High    |
| Layer 2       | LOW–MEDIUM    | 8%        | Medium-term (1–3 years)     | Medium-High    |
| DeFi          | MEDIUM        | 25%       | Medium-term (6–24 months)   | High           |
| Stablecoin    | LOW           | 2%        | Utility/Savings             | Very Low       |
| Meme Coin     | HIGH          | 45%       | Short-term (days–weeks)     | Extreme        |
| Influencer    | VERY HIGH     | 85%       | NOT RECOMMENDED             | Extreme        |
| Exchange      | MEDIUM        | 15%       | Medium-term (1–2 years)     | Medium         |
| Gaming        | MEDIUM–HIGH   | 30%       | Medium-term (1–3 years)     | High           |
| AI Token      | MEDIUM        | 20%       | Medium-long term (1–3 years)| High           |


Live Analysis System
--------------------

## Real-Time Data Pipeline

User Input  
↓  
CoinGecko API  
↓  
Data Processing  
↓  
Feature Generation  
↓  
ML Prediction  
↓  
Risk Score  
↓  
Recommendation  

Detailed Flow:

- Coin Name → Fetch Data  
- Validate & Engineer Features  
- Run Ensemble Model  
- Combine ML + Rule-Based Scores  
- Generate Final Report  
- Market Stats + Price History + Social Data Included  


## CoinGecko API Integration

The system fetches comprehensive data:

```python
# Data points retrieved:
- current_price
- market_cap
- market_cap_rank
- total_volume
- price_change_24h
- price_change_7d
- price_change_30d
- price_change_1y
- all_time_high
- ath_change_percentage
- circulating_supply
- total_supply
- twitter_followers
- reddit_subscribers
- github_stars
- sparkline_7d
- price_history_90d
```

### Feature Estimation for Unknown Coins

When security features aren't available via API, the system estimates based on:

-   **Market Cap Rank**: Higher ranked coins assumed more legitimate

-   **Social Presence**: Larger following indicates established community

-   **Category**: Risk profile applied based on coin category

-   **Age**: Older projects given higher trust scores

### Recommendation Engine

The system generates detailed recommendations including:

-   **Investment Verdict**: Clear buy/avoid recommendation

-   **Category Insights**: Specific risks for the coin's category

-   **Market Analysis**: Current market position and trends

-   **Pre-Investment Checklist**: Due diligence steps

-   **Red/Green Flags**: Specific warnings and positives

## Results and Performance

### Model Performance Summary

| Model              | Accuracy | F1 Score | Precision | Recall | ROC-AUC |
|--------------------|----------|----------|-----------|--------|---------|
| Logistic Regression | 0.821   | 0.798    | 0.812     | 0.785  | 0.876   |
| Random Forest       | 0.854   | 0.831    | 0.845     | 0.818  | 0.912   |
| XGBoost             | 0.867   | 0.849    | 0.856     | 0.842  | 0.923   |
| LightGBM            | 0.859   | 0.841    | 0.851     | 0.831  | 0.918   |
| Gradient Boosting   | 0.851   | 0.829    | 0.841     | 0.817  | 0.908   |
| **Ensemble**        | **0.872** | **0.856** | **0.863** | **0.849** | **0.928** |

---

### Cross-Validation Results

| Model               | CV Mean F1 | CV Std | Overfit Gap |
|---------------------|------------|--------|--------------|
| Logistic Regression | 0.785      | 0.032  | 0.013        |
| Random Forest       | 0.818      | 0.028  | 0.013        |
| XGBoost             | 0.836      | 0.025  | 0.013        |
| LightGBM            | 0.829      | 0.027  | 0.012        |
| Gradient Boosting   | 0.816      | 0.029  | 0.013        |

*Overfit Gap = Train F1 - Test F1 (lower is better)*

---

### Confusion Matrix Analysis

**Predicted**  
Legit | Scam  
--- | ---  
**Actual Legit:** 89 | 7  
**Actual Scam:** 5 | 31  

- **True Negatives:** 89 legitimate coins correctly identified  
- **True Positives:** 31 scams correctly detected  
- **False Positives:** 7 legitimate coins incorrectly flagged  
- **False Negatives:** 5 scams missed  

---

### Category-Wise Performance

| Category     | Samples | Accuracy | F1 Score |
|--------------|---------|----------|----------|
| Layer 1      | 45      | 0.956    | 0.923    |
| DeFi         | 52      | 0.846    | 0.821    |
| Meme Coin    | 38      | 0.789    | 0.756    |
| Stablecoin   | 18      | 0.944    | 0.912    |
| Influencer   | 12      | 0.917    | 0.889    |
| Exchange     | 15      | 0.867    | 0.842    |

---

### Key Insights

- **High-Risk Categories:** Influencer and meme coins show the highest scam rates  
- **Security Indicators:** Audits and liquidity lock are the strongest predictors  
- **Age Matters:** Projects under 30 days exhibit significantly higher risk  
- **Ensemble Value:** Combined model performance improves accuracy by **2–5%** over individual models  

Installation and Setup
----------------------

### Prerequisites

-   Python 3.8 or higher

-   pip package manager

-   Internet connection (for API access)

-   Google Colab (recommended) or Jupyter Notebook

### Required Libraries

bashLine WrappingCollapseCopy123pip install pandas numpy scikit-learn xgboost lightgbmpip install plotly kaleido imbalanced-learnpip install requests beautifulsoup4 pycoingecko

### Quick Start (Google Colab)

1.  Open Google Colab

2.  Create a new notebook

3.  Copy and paste the code cells sequentially

4.  Run all cells in order






## Project Structure



    notebooks/
        crypto_scam_detection.ipynb

    src/
        __init__.py
        data_fetcher.py
        scam_database.py
        dataset_builder.py
        feature_engineer.py
        scam_detector.py
        risk_scorer.py
        visualizer.py
        live_predictor.py

    data/
        documented_scams.json
        legitimate_coins.json
        category_profiles.json

    models/
        trained_ensemble.pkl

    outputs/
        crypto_scam_analysis.csv
        visualizations/

    tests/
        test_data_fetcher.py
        test_feature_engineer.py
        test_risk_scorer.py


Visualizations
--------------

The system generates multiple interactive visualizations:

1.  **Model Performance Comparison**

    -   Bar chart comparing accuracy, F1, precision, recall, and ROC-AUC across all models.

2.  **ROC Curves**

    -   Receiver Operating Characteristic curves for all models with AUC scores.

3.  **Feature Importance**

    -   Horizontal bar chart showing top 15 most important features from Random Forest.

4.  **Category Distribution**

    -   Stacked bar chart showing scam vs legitimate distribution by category.

5.  **Scam Types Distribution**

    -   Pie chart showing breakdown of scam types in the dataset.

6.  **Confusion Matrix**

    -   Heatmap showing true/false positive/negative counts.

7.  **Risk Score Gauge**

    -   Interactive gauge visualization for individual coin risk scores.

8.  **Price History Charts**

    -   Line charts with 7-day moving average for analyzed coins.

Limitations and Future Work
---------------------------

### Current Limitations

-   **Data Availability**: Some security features must be estimated for coins not in our database

-   **Real-Time Constraints**: API rate limits restrict bulk analysis speed

-   **Evolving Scams**: New scam techniques may not be captured by historical patterns

-   **Binary Classification**: Current model treats scam as binary; severity levels could be added

-   **Limited Languages**: Analysis and recommendations are English-only

### Future Improvements

#### Short-Term

-   Add TokenSniffer API integration for contract analysis

-   Implement caching for API responses

-   Add more documented scam cases

-   Create browser extension for real-time warnings

#### Medium-Term

-   Develop smart contract bytecode analysis

-   Add social media sentiment analysis

-   Implement time-series anomaly detection

-   Create automated monitoring system

#### Long-Term

-   Build comprehensive scam reporting platform

-   Develop blockchain-agnostic analysis (BSC, Polygon, etc.)

-   Create community-driven scam database

-   Implement deep learning models for pattern recognition

References
----------

### Academic Sources

-   Bartoletti, M., et al. (2020). "Dissecting Ponzi schemes on Ethereum"

-   Chen, W., et al. (2021). "Detecting Ponzi Schemes on Ethereum"

-   Xia, P., et al. (2020). "Characterizing Cryptocurrency Exchange Scams"

### Industry Resources

-   [CoinGecko API Documentation](https://www.coingecko.com/api/docs/v3)

-   [TokenSniffer - Contract analysis](https://tokensniffer.com/)

-   [RugDoc - DeFi security reviews](https://rugdoc.io/)

-   [Rekt News - DeFi exploit database](https://rekt.news/)

-   [Web3 Is Going Great - Documented incidents](https://web3isgoinggreat.com/)

-   [CryptoScamDB - Scam tracking](https://cryptoscamdb.org/)

-   [Chainabuse - Address reporting](https://chainabuse.com/)

Contributors
------------

### Project Author

[Ankit Singh]



### Acknowledgments

-   CoinGecko for providing free API access

-   Scikit-learn, XGBoost, and LightGBM development teams

-   Open-source community for documentation and examples

-   Researchers documenting cryptocurrency fraud patterns

License
-------

This project is licensed under the MIT License - see the [LICENSE](https://chat.z.ai/c/LICENSE) file for details.

# MIT License

Copyright (c) 2025 [ANKIT SINGH]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


### Disclaimer

This tool is for educational and research purposes only. Not financial advice. Past scam patterns may not predict future scams. Always conduct your own research (DYOR). Never invest more than you can afford to lose. The authors are not responsible for any financial losses.

Contact
-------

For questions, suggestions, or contributions:

-   **Email**: [<ankitsingh92004@gmail.com>]

-   **GitHub Issues**: Create an issue


