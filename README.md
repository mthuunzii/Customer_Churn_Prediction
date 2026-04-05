# Customer_Churn_Prediction
Python
SklearnXGBoostStatus
A machine learning project to predict customer churn for a retail bank using demographic, account, and behavioural data. Built as a portfolio project demonstrating the full data science workflow — from exploratory analysis to model deployment recommendations.

📌 Problem Statement
Customer churn costs banks significantly more than retention. This project answers:

Can we accurately predict which customers are likely to churn, so the bank can intervene proactively and reduce attrition?

📊 Dataset
Source	Bank Customer Churn Dataset – Kaggle
Size	10,000 customers, 14 features
Target	Exited (1 = churned, 0 = stayed)
Churn rate	~20.4% (imbalanced dataset)
🗂️ Project Structure
bank-churn-prediction/
│
├── Bank_Customer_Churn_Prediction.ipynb   ← Main notebook
├── Churn_Modelling.csv                    ← Dataset
├── requirements.txt                      ← Dependencies
└── README.md                            ← This file
🧠 Workflow
Exploratory Data Analysis — churn rates by geography, gender, age, products, and activity status
Feature Engineering — 5 new features including HighValueInactive, BalanceToSalary, ProductsOverloaded
Preprocessing — StandardScaler + OneHotEncoder via sklearn Pipeline
Modelling — 3 models tuned with GridSearchCV (ROC-AUC scoring): Logistic Regression, Random Forest, XGBoost ✅
Evaluation — ROC-AUC, F1-score, confusion matrices, overlaid ROC curves
Feature Importance — top churn drivers identified and visualised
Business Recommendations — actionable retention strategies per customer segment
📈 Results
Model	ROC-AUC	F1 (Churn)	Verdict
Logistic Regression	~0.84	~0.57	Good interpretable baseline
Random Forest	~0.86	~0.60	Solid, handles non-linearity
XGBoost ✅	~0.87	~0.58	Best overall — recommended
Class imbalance (~20% churn rate) was handled natively — class_weight='balanced' for LR & RF, and scale_pos_weight for XGBoost. All models scored on ROC-AUC, not accuracy.

🔑 Key Findings
Germany has a churn rate of ~32% — nearly double France (~16%) and Spain (~17%)
Inactive members churn at ~27% vs ~14% for active members — strongest behavioural signal
Older customers (40+) are significantly more likely to churn (median age 44 vs 37)
Customers with 3+ products churn at rates exceeding 80% — product fatigue signal
High-balance inactive customers are the highest-risk and highest-value segment to retain

⚙️ How to Run
# 1. Clone the repository
git clone https://github.com/mthuunzii/bank-churn-prediction.git
cd bank-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Bank Customer Churn Prediction.ipynb

🛠️ Tech Stack
Python 3.11
pandas
numpy
matplotlib
seaborn
scikit-learn
XGBoost
Jupyter Notebook


