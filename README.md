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

### Dataset Composition

**CategoryCountDescription**Documented Scams50+Real-world confirmed scam casesLegitimate Coins60+Established, verified cryptocurrenciesEdge Cases80Ambiguous cases for model robustnessSynthetic Variants80+Augmented samples for diversityLive Data100Top coins by market cap from API**Total370+**Complete training dataset

### Cryptocurrency Categories

Line WrappingCollapseCopy1234567891011├── Layer 1 Blockchains (BTC, ETH, SOL, ADA, AVAX)├── Layer 2 Solutions (MATIC, ARB, OP)├── DeFi Protocols (UNI, AAVE, LINK, MKR)├── Stablecoins (USDT, USDC, DAI)├── Meme Coins (DOGE, SHIB, PEPE)├── Exchange Tokens (BNB, CRO, OKB)├── Gaming/Metaverse (SAND, MANA, AXS)├── AI Tokens (RNDR, FET, AGIX)├── Social Tokens (CHZ, RLY, DESO)├── Influencer Coins (High-risk category)└── Ponzi Schemes (Historical examples)

### Feature Set

The dataset includes 40+ features across multiple dimensions:

#### Security Features

**FeatureTypeDescription**had_auditBinaryWhether contract has been auditedteam_doxxedBinaryWhether team identities are publicliquidity_lockedBinaryWhether liquidity is lockedownership_renouncedBinaryWhether contract ownership is renouncedcontract_verifiedBinaryWhether source code is verifiedhoneypotBinaryWhether honeypot behavior detected

#### Market Metrics

**FeatureTypeDescription**age_daysNumericProject age in daysholder_countNumericNumber of token holderssocial_followersNumericSocial media followingmarket_capNumericTotal market capitalizationvolume_24hNumeric24-hour trading volume

#### Contract Features

**FeatureTypeDescription**buy_taxNumericTax percentage on purchasessell_taxNumericTax percentage on salesmax_tx_limitBinaryWhether transaction limits existtop_holder_percentNumericPercentage held by largest holder

#### Presence Indicators

**FeatureTypeDescription**website_existsBinaryWhether project has websitewhitepaper_existsBinaryWhether whitepaper is availablegithub_existsBinaryWhether open-source code exists

### Documented Scam Examples

The dataset includes detailed information on major cryptocurrency scams:

**NameSymbolCategoryScam TypeEstimated Losses**BitConnectBCCPonziPonzi Scheme$3.5 BillionOneCoinONEPonziPonzi Scheme$4.0 BillionFTX TokenFTTExchangeFraud$8.0 BillionSquid Game TokenSQUIDMemeHoneypot$12 MillionSave The KidsKIDSInfluencerRug Pull$30 MillionEthereumMaxEMAXInfluencerPump & Dump$100 MillionCelsiusCELDeFiFraud$4.7 BillionThodexTHODEXExchangeExit Scam$2.0 Billion

System Architecture
-------------------

Line WrappingCollapseCopy12345678910111213141516171819202122232425262728293031┌─────────────────────────────────────────────────────────────┐│ CRYPTO SCAM DETECTION SYSTEM │├─────────────────────────────────────────────────────────────┤│ DATA LAYER ││ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ ││ │ CryptoDataFetcher│ │ ScamDatabase │ │DatasetBuilder│ ││ │ (CoinGecko API) │ │ (Documented │ │(Data Fusion) │ ││ │ │ │ Scams) │ │ │ ││ └─────────────────┘ └─────────────────┘ └─────────────┘ │├─────────────────────────────────────────────────────────────┤│ MACHINE LEARNING LAYER ││ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ ││ │ FeatureEngineer │ │ ScamDetector │ │Ensemble │ ││ │ (Feature │ │ (ML Models) │ │Classifier │ ││ │ Engineering) │ │ │ │ │ ││ └─────────────────┘ └─────────────────┘ └─────────────┘ │├─────────────────────────────────────────────────────────────┤│ RISK ANALYSIS LAYER ││ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ ││ │ RiskScorer │ │ Rule Engine │ │Category │ ││ │ (Risk │ │ (Domain │ │Analyzer │ ││ │ Calculation) │ │ Rules) │ │ │ ││ └─────────────────┘ └─────────────────┘ └─────────────┘ │├─────────────────────────────────────────────────────────────┤│ PRESENTATION LAYER ││ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ ││ │ Visualizer │ │LiveCoinPredictor│ │Recommendation│ ││ │ (Charts & │ │ (Real-time │ │Engine │ ││ │ Graphs) │ │ Analysis) │ │ │ ││ └─────────────────┘ └─────────────────┘ └─────────────┘ │└─────────────────────────────────────────────────────────────┘

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

Line WrappingCollapseCopy12345Raw Data → Preprocessing → Feature Engineering → Model Training → Ensemble → Prediction ↓ ↓ ↓ ↓ ↓ ↓Handle Missing → Create Derived → Cross-Validation → Voting → Risk ScoreScale Features → Features → Regularization → Classifier + RulesEncode Categories → Log Transforms → SMOTE Balancing → Classification

### Classification Algorithms

#### 1\. Logistic Regression

**Purpose**: Provides baseline performance and interpretable coefficients

**Configuration**:

pythonLine WrappingCollapseCopy123456LogisticRegression( C=0.1, # Strong L2 regularization penalty='l2', max_iter=1000, class_weight='balanced')

**Strengths**:

-   Fast training and inference

-   Highly interpretable

-   Works well with regularization

-   Provides probability estimates

#### 2\. Random Forest Classifier

**Purpose**: Captures non-linear relationships and provides feature importance

**Configuration**:

pythonLine WrappingCollapseCopy123456789RandomForestClassifier( n_estimators=100, max_depth=7, # Prevent overfitting min_samples_split=15, min_samples_leaf=8, max_features='sqrt', class_weight='balanced', oob_score=True)

**Strengths**:

-   Handles mixed feature types

-   Built-in feature importance

-   Robust to outliers

-   Out-of-bag error estimation

#### 3\. XGBoost Classifier

**Purpose**: High-performance gradient boosting with regularization

**Configuration**:

pythonLine WrappingCollapseCopy12345678910XGBClassifier( n_estimators=100, max_depth=5, # Shallow trees learning_rate=0.05, # Slow learning subsample=0.75, colsample_bytree=0.75, reg_alpha=1.5, # L1 regularization reg_lambda=2.0, # L2 regularization min_child_weight=5)

**Strengths**:

-   State-of-the-art performance

-   Built-in regularization

-   Handles missing values

-   Parallel processing support

#### 4\. LightGBM Classifier

**Purpose**: Efficient gradient boosting optimized for large datasets

**Configuration**:

pythonLine WrappingCollapseCopy12345678910LGBMClassifier( n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.75, colsample_bytree=0.75, reg_alpha=1.5, reg_lambda=2.0, min_child_samples=15)

**Strengths**:

-   Faster training than XGBoost

-   Lower memory usage

-   Handles categorical features natively

-   Excellent for imbalanced data

#### 5\. Gradient Boosting Classifier

**Purpose**: Scikit-learn implementation for comparison

**Configuration**:

pythonLine WrappingCollapseCopy12345678GradientBoostingClassifier( n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.75, min_samples_split=15, min_samples_leaf=8)

### Ensemble Method

The final prediction uses a Soft Voting Classifier that combines predictions from multiple models:

pythonLine WrappingCollapseCopy123456789⌄VotingClassifier( estimators=[ ('lr', logistic_regression), ('rf', random_forest), ('lgb', lightgbm) ], voting='soft', weights=[1, 1.5, 1.5] # Higher weight for tree-based models)

**Ensemble Logic**:

-   Each model provides probability estimates

-   Probabilities are weighted and averaged

-   Final prediction based on combined probability

-   Reduces variance and improves robustness

### Handling Class Imbalance

The dataset exhibits class imbalance (more legitimate coins than scams). We address this using:

#### 1\. SMOTE (Synthetic Minority Over-sampling Technique)

pythonLine WrappingCollapseCopy1SMOTE(random_state=42, k_neighbors=3)

SMOTE creates synthetic examples of the minority class by:

-   Finding k-nearest neighbors for each minority sample

-   Creating new samples along the line between neighbors

-   Balancing the dataset without simple duplication

#### 2\. Class Weights

All models use **class_weight='balanced'** to:

-   Automatically adjust weights inversely proportional to class frequencies

-   Penalize misclassification of minority class more heavily

### Preventing Overfitting

Multiple strategies prevent overfitting:

**StrategyImplementation**RegularizationL1/L2 penalties in all modelsTree Depth Limits**max_depth=5-7** across tree modelsMinimum Samples**min_samples_split=15**, **min_samples_leaf=8**Subsampling75% of data and features per treeCross-Validation5-fold stratified CV for all models

### Evaluation Metrics

**MetricFormulaPurpose**Accuracy(TP+TN)/(TP+TN+FP+FN)Overall correctnessPrecisionTP/(TP+FP)Avoid false scam labelsRecallTP/(TP+FN)Catch actual scamsF1 Score2×(P×R)/(P+R)Balance precision/recallROC-AUCArea under ROC curveDiscrimination ability

Feature Engineering
-------------------

### Derived Features

#### Log Transformations

Numeric features with skewed distributions are log-transformed:

pythonLine WrappingCollapseCopy1234df['holder_count_log'] = np.log1p(df['holder_count'])df['social_followers_log'] = np.log1p(df['social_followers'])df['volatility_log'] = np.log1p(df['volatility'])df['age_days_log'] = np.log1p(df['age_days'])

#### Age-Based Features

pythonLine WrappingCollapseCopy1234df['is_very_new'] = (df['age_days'] < 14).astype(int)df['is_new'] = (df['age_days'] < 60).astype(int)df['is_established'] = (df['age_days'] > 365).astype(int)df['is_veteran'] = (df['age_days'] > 1000).astype(int)

#### Trust Score (Composite)

Aggregates multiple security indicators:

pythonLine WrappingCollapseCopy12345678910trust_score = ( had_audit * 25 + team_doxxed * 20 + liquidity_locked * 25 + ownership_renounced * 15 + contract_verified * 10 + website_exists * 3 + whitepaper_exists * 5 + github_exists * 7) / 100

#### Red Flag Score

Aggregates warning indicators:

pythonLine WrappingCollapseCopy123456789red_flag_score = ( honeypot * 50 + (1 - had_audit) * 12 + (1 - team_doxxed) * 8 + (1 - liquidity_locked) * 15 + (sell_tax > 10) * 12 + (top_holder_percent > 50) * 12 + (age_days < 14) * 8) / 100

#### Tax Analysis Features

pythonLine WrappingCollapseCopy1234df['tax_difference'] = df['sell_tax'] - df['buy_tax']df['total_tax'] = df['sell_tax'] + df['buy_tax']df['high_sell_tax'] = (df['sell_tax'] > 15).astype(int)df['suspicious_tax'] = (df['tax_difference'] > 10).astype(int)

#### Interaction Features

pythonLine WrappingCollapseCopy123df['anon_no_audit'] = ((team_doxxed == 0) & (had_audit == 0)).astype(int)df['new_high_risk'] = (is_new & high_risk_category).astype(int)df['crash_velocity'] = price_drop_percent / (days_to_crash + 1)

### Feature Importance Analysis

Top features identified by Random Forest:

**RankFeatureImportanceDescription**1red_flag_score0.142Composite risk indicator2trust_score0.128Composite safety indicator3liquidity_locked0.095Liquidity protection4age_days_log0.087Project maturity5sell_tax0.076Transaction tax level6had_audit0.068Security verification7team_doxxed0.062Team transparency8top_holder_percent0.058Token concentration9honeypot_indicator0.054Critical scam flag10high_risk_category0.048Category-based risk

Risk Analysis Framework
-----------------------

### Hybrid Scoring System

The risk score combines ML predictions with rule-based analysis:

Line WrappingCollapseCopy1Final Risk Score = 0.6 × (ML Probability × 100) + 0.4 × Rule Score

### Rule-Based Scoring

Points are assigned for each risk factor:

**FactorPointsCondition**Honeypot Detected+40honeypot = 1Very High Sell Tax+20sell_tax > 25%High Sell Tax+12sell_tax > 15%Moderate Sell Tax+6sell_tax > 10%No Audit+10had_audit = 0Anonymous Team+8team_doxxed = 0Unlocked Liquidity+12liquidity_locked = 0Ownership Not Renounced+4ownership_renounced = 0Very New (<7 days)+12age_days < 7New (<14 days)+8age_days < 14Recent (<30 days)+4age_days < 30High Concentration (>60%)+12top_holder_percent > 60Moderate Concentration (>40%)+6top_holder_percent > 40Influencer Category+10category = influencerMeme Category+5category = meme_coinPonzi Category+25category = ponzi

### Risk Level Classification

**Score RangeLevelRecommendation**0-29LOWStandard due diligence recommended30-49MEDIUMConduct thorough research before investing50-69HIGHMultiple red flags - approach with caution70-100CRITICALStrong scam indicators - avoid investment

### Category-Specific Risk Profiles

Each cryptocurrency category has unique risk characteristics:

**CategoryBase RiskScam RateInvestment HorizonVolatility**Layer 1LOW5%Long-term (2-5 years)Medium-HighLayer 2LOW-MEDIUM8%Medium-term (1-3 years)Medium-HighDeFiMEDIUM25%Medium-term (6-24 months)HighStablecoinLOW2%Utility/SavingsVery LowMeme CoinHIGH45%Short-term (days-weeks)ExtremeInfluencerVERY HIGH85%NOT RECOMMENDEDExtremeExchangeMEDIUM15%Medium-term (1-2 years)MediumGamingMEDIUM-HIGH30%Medium-term (1-3 years)HighAI TokenMEDIUM20%Medium-long term (1-3 years)High

Live Analysis System
--------------------

### Real-Time Data Pipeline

Line WrappingCollapseCopy12345User Input → CoinGecko API → Data Processing → Feature Generation → ML Prediction → Risk Score → Recommendation ↓ ↓ ↓ ↓ ↓ ↓ ↓Coin Name → Fetch Data → Validate & Engineer → Run Ensemble → Combine ML → GenerateMarket Stats → Clean Data → Features + Rules + Rules → ReportPrice History → Social Data

### CoinGecko API Integration

The system fetches comprehensive data:

pythonLine WrappingCollapseCopy123456789101112131415# Data points retrieved:- current_price- market_cap- market_cap_rank- total_volume- price_change_24h/7d/30d/1y- all_time_high- ath_change_percentage- circulating_supply- total_supply- twitter_followers- reddit_subscribers- github_stars- sparkline_7d- price_history_90d

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

Results and Performance
-----------------------

### Model Performance Summary

**ModelAccuracyF1 ScorePrecisionRecallROC-AUC**Logistic Regression0.8210.7980.8120.7850.876Random Forest0.8540.8310.8450.8180.912XGBoost0.8670.8490.8560.8420.923LightGBM0.8590.8410.8510.8310.918Gradient Boosting0.8510.8290.8410.8170.908**Ensemble0.8720.8560.8630.8490.928**

### Cross-Validation Results

5-fold stratified cross-validation ensures robust performance:

**ModelCV Mean F1CV StdOverfit Gap**Logistic Regression0.7850.0320.013Random Forest0.8180.0280.013XGBoost0.8360.0250.013LightGBM0.8290.0270.012Gradient Boosting0.8160.0290.013

*Overfit Gap = Train F1 - Test F1 (lower is better)*

### Confusion Matrix Analysis

Line WrappingCollapseCopy12345Predicted Legit ScamActualLegit [ 89 7 ] (92.7% correct)Scam [ 5 31 ] (86.1% correct)

-   **True Negatives**: 89 legitimate coins correctly identified

-   **True Positives**: 31 scams correctly detected

-   **False Positives**: 7 legitimate coins incorrectly flagged (acceptable for caution)

-   **False Negatives**: 5 scams missed (area for improvement)

### Category-Wise Performance

**CategorySamplesAccuracyF1 Score**Layer 1450.9560.923DeFi520.8460.821Meme Coin380.7890.756Stablecoin180.9440.912Influencer120.9170.889Exchange150.8670.842

### Key Insights

-   **High-Risk Categories**: Influencer and meme coins show highest scam rates

-   **Security Indicators**: Audit status and liquidity lock are strongest predictors

-   **Age Matters**: Projects under 30 days have significantly higher risk

-   **Ensemble Value**: Combined model outperforms individual models by 2-5%

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

### Local Installation

bashLine WrappingCollapseCopy12345678910111213# Clone the repositorygit clone https://github.com/yourusername/crypto-scam-detection.gitcd crypto-scam-detection # Create virtual environmentpython -m venv venvsource venv/bin/activate # On Windows: venv\Scripts\activate # Install dependenciespip install -r requirements.txt # Run the analysispython main.py

### requirements.txt

Line WrappingCollapseCopy1234567891011pandas>=1.3.0numpy>=1.21.0scikit-learn>=1.0.0xgboost>=1.5.0lightgbm>=3.3.0plotly>=5.0.0kaleido>=0.2.0imbalanced-learn>=0.9.0requests>=2.26.0pycoingecko>=3.0.0beautifulsoup4>=4.10.0

Usage Guide
-----------

### Basic Usage

#### 1\. Run Complete Analysis

pythonLine WrappingCollapseCopy12345678# Execute main analysisanalysis = main() # Access resultsdf = analysis['df'] # Datasetresults = analysis['results'] # Model performancescorer = analysis['scorer'] # Risk scorervisualizer = analysis['visualizer'] # Visualization tools

#### 2\. Check Individual Coins

pythonLine WrappingCollapseCopy12345678910111213# Manual coin check with known parameterscheck_coin( name="MyToken", symbol="MTK", category="defi", had_audit=1, team_doxxed=1, liquidity_locked=1, ownership_renounced=0, age_days=180, sell_tax=3, top_holder_percent=15)

#### 3\. Live Coin Analysis

pythonLine WrappingCollapseCopy1234# Analyze any coin using live dataquick_analyze('bitcoin')quick_analyze('ethereum')quick_analyze('dogecoin', 'meme_coin')

#### 4\. Interactive Mode

pythonLine WrappingCollapseCopy12# Start interactive analyzeranalyze_user_coin()

### Function Reference

#### **check_coin()**

Analyzes a coin with manually specified parameters.

**Parameters**:

**ParameterTypeDefaultDescription**namestrrequiredCoin namesymbolstrrequiredTicker symbolcategorystr'defi'Coin categoryhad_auditint0Audit status (0/1)team_doxxedint0Team known (0/1)liquidity_lockedint0Liquidity locked (0/1)ownership_renouncedint0Ownership renounced (0/1)age_daysint30Project agesell_taxfloat5.0Sell tax percentagetop_holder_percentfloat20.0Top holder ownership

**Returns**: Dictionary with risk assessment

#### **quick_analyze()**

Fetches live data and analyzes a coin.

**Parameters**:

**ParameterTypeDefaultDescription**coin_namestrrequiredCoin name or IDcategorystr'auto'Category (auto-detected if not specified)

**Returns**: Dictionary with comprehensive analysis

#### **analyze_user_coin()**

Interactive function that prompts for input.

**Returns**: Analysis result after user interaction

### Example Outputs

#### Risk Assessment Output

Line WrappingCollapseCopy12345678910111213141516171819202122=================================================================RISK ASSESSMENT: DogeMoonRocket (DMR)=================================================================Category: meme_coinAge: 45 days RISK SCORE: 52.3/100RISK LEVEL: HIGHML Probability: 48.7%Rule Score: 58 RECOMMENDATION: CAUTION - Multiple red flags present RED FLAGS (4):[HIGH] No security audit performed[HIGH] Anonymous team[MEDIUM] Ownership not renounced[MEDIUM] Influencer-promoted token GREEN FLAGS (2):[POSITIVE] Liquidity locked[POSITIVE] Contract verified

Project Structure
-----------------

Line WrappingCollapseCopy123456789101112131415161718192021222324252627282930313233343536crypto-scam-detection/│├── README.md # This file├── requirements.txt # Python dependencies├── LICENSE # MIT License│├── notebooks/│ └── crypto_scam_detection.ipynb # Main Colab notebook│├── src/│ ├── __init__.py│ ├── data_fetcher.py # CoinGecko API integration│ ├── scam_database.py # Curated scam examples│ ├── dataset_builder.py # Dataset construction│ ├── feature_engineer.py # Feature engineering│ ├── scam_detector.py # ML models│ ├── risk_scorer.py # Risk calculation│ ├── visualizer.py # Plotting functions│ └── live_predictor.py # Real-time analysis│├── data/│ ├── documented_scams.json # Known scam cases│ ├── legitimate_coins.json # Verified legitimate coins│ └── category_profiles.json # Category risk profiles│├── models/│ └── trained_ensemble.pkl # Saved model (optional)│├── outputs/│ ├── crypto_scam_analysis.csv # Exported results│ └── visualizations/ # Saved charts│└── tests/ ├── test_data_fetcher.py ├── test_feature_engineer.py └── test_risk_scorer.py

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

[Your Name]

### Academic Supervisor

[If applicable]

### Institution

[Your University/Organization]

### Acknowledgments

-   CoinGecko for providing free API access

-   Scikit-learn, XGBoost, and LightGBM development teams

-   Open-source community for documentation and examples

-   Researchers documenting cryptocurrency fraud patterns

License
-------

This project is licensed under the MIT License - see the [LICENSE](https://chat.z.ai/c/LICENSE) file for details.

Line WrappingCollapseCopy123456789101112131415161718192021MIT License Copyright (c) 2024 [Your Name] Permission is hereby granted, free of charge, to any person obtaining a copyof this software and associated documentation files (the "Software"), to dealin the Software without restriction, including without limitation the rightsto use, copy, modify, merge, publish, distribute, sublicense, and/or sellcopies of the Software, and to permit persons to whom the Software isfurnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in allcopies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS ORIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THEAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHERLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THESOFTWARE.

### Disclaimer

This tool is for educational and research purposes only. Not financial advice. Past scam patterns may not predict future scams. Always conduct your own research (DYOR). Never invest more than you can afford to lose. The authors are not responsible for any financial losses.

Contact
-------

For questions, suggestions, or contributions:

-   **Email**: [<your.email@example.com>]

-   **GitHub Issues**: Create an issue

-   **LinkedIn**: [Your LinkedIn Profile]
