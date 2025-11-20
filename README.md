# Interpretable-Machine-learning-for-Credit-Risk-Modeling-using-SHAP-Values

1) Problem statement
Predict whether a borrower will default (binary classification) using the UCI Statlog (German Credit) dataset. The model must be accurate and fully interpretable so underwriting teams can understand and act on model outputs.
2) Dataset & target
Dataset: UCI Statlog (German Credit), 1000 records, 21 features (account status, duration, credit history, purpose, savings, employment, amount, age, etc.).
Target mapping used: UCI 1 = Good → 0; 2 = Bad → 1 (default).
3) Approach (short)
Preprocessing: numeric scaling (StandardScaler) and OneHotEncoding for categorical features via ColumnTransformer.
Imbalance handling: SMOTE on training set.
Model: LightGBM tuned by RandomizedSearchCV with stratified 4-fold CV.
Calibration: Platt scaling (CalibratedClassifierCV).
Threshold selection: chosen on validation to maximize F1.
Explainability: SHAP global + local, LIME local; all local explanations transcribed in this submission.
4) Model performance (actual numbers from the run)
Validation AUC: 0.7445502646
Validation F1: 0.4634
Test AUC: 0.7864550265
Test F1: 0.4935
Test Precision: 0.5938
Test Recall: 0.4222
(ROC, PR, confusion matrix images are included in the repo artifacts.)
5) Global feature importance (top 5)
Computed from model explanations (SHAP or permutation importance fallback):
Status_checking_account
Credit_history
Credit_amount
Savings_account_bonds
Purpose
Full CSV: global_shap_all.csv.
6) Local SHAP explanations — 5 profiles (transcribed)
Each profile below shows: Profile ID, predicted probability, true label, top SHAP contributions (feature : SHAP value — positive increases default risk; negative decreases), and a 1–2 line plain-English summary.
Profile ID: 29
Predicted probability (default): 0.7056
True label: 1 (default)
Top SHAP contributions:
Status_checking_account_A14 : +0.2742 (increases risk)
Credit_amount : +0.1558 (increases risk)
Duration_months : +0.1108 (increases risk)
Savings_account_bonds_A61 : +0.0923 (increases risk)
Installment_rate : +0.0895 (increases risk)
Status_checking_account_A12 : +0.0819 (increases risk)
Status_checking_account_A11 : -0.0556 (decreases risk)
Property_A124 : +0.0469
Property_A121 : +0.0436
Credit_history_A34 : +0.0402
Plain English summary: Large checking-account flag (A14), a high credit amount and longer duration are the main drivers pushing this applicant into high risk. Some checking-account categories slightly reduce risk, but not enough to offset large positive contributors.
Profile ID: 74
Predicted probability (default): 0.6898
True label: 1 (default)
Top SHAP contributions:
Status_checking_account_A14 : +0.2615
Status_checking_account_A11 : +0.2178
Duration_months : +0.1775
Installment_rate : +0.0954
Purpose_A43 : -0.0715 (decreases risk)
Savings_account_bonds_A61 : +0.0677
Credit_history_A34 : +0.0476
Housing_A152 : +0.0428
Property_A124 : +0.0365
Property_A122 : +0.0236
Plain English summary: Multiple checking-account indicators and longer loan duration strongly increase risk. One purpose category (A43) slightly reduces risk, but overall contributions indicate high default probability.
Profile ID: 135
Predicted probability (default): 0.6891
True label: 1 (default)
Top SHAP contributions:
Status_checking_account_A14 : +0.2550
Status_checking_account_A11 : +0.1907
Duration_months : +0.1379
Savings_account_bonds_A61 : +0.1213
Installment_rate : +0.0918
Credit_history_A34 : +0.0664
Property_A121 : +0.0365
Purpose_A42 : -0.0347 (decreases risk)
Other_installment_plans_A141 : -0.0280 (decreases risk)
Property_A122 : +0.0247
Plain English summary: Similar to other high-risk profiles: account status and duration are dominant risk drivers, with some features (certain purpose and installment plan values) slightly reducing risk.
Profile ID: 143
Predicted probability (default): 0.2392
True label: 0 (no default)
Top SHAP contributions:
Status_checking_account_A14 : -0.3670 (decreases risk)
Duration_months : -0.2118 (decreases risk)
Credit_history_A34 : -0.0947 (decreases risk)
Status_checking_account_A11 : -0.0632 (decreases risk)
Other_installment_plans_A141 : -0.0526 (decreases risk)
Age_years : -0.0504 (decreases risk)
Property_A121 : -0.0451 (decreases risk)
Status_checking_account_A12 : -0.0399 (decreases risk)
Installment_rate : -0.0354 (decreases risk)
Credit_amount : -0.0329 (decreases risk)
Plain English summary: Strong negative SHAP values across checking-account status, duration, and credit history indicate this applicant is low risk; these features push prediction well below the default threshold.
Profile ID: 100
Predicted probability (default): 0.2394
True label: 0 (no default)
Top SHAP contributions:
Status_checking_account_A14 : -0.3049
Savings_account_bonds_A61 : -0.1742
Credit_history_A34 : -0.1006
Status_checking_account_A11 : -0.0921
Other_installment_plans_A141 : -0.0668
Savings_account_bonds_A65 : -0.0637
Age_years : -0.0398
Purpose_A41 : -0.0290
Other_installment_plans_A143 : -0.0256
Personal_status_sex_A92 : -0.0252
Plain English summary: Multiple strong negative contributors (checking account, savings, credit history) combine to give a confident low-risk prediction.
7) Local LIME explanations — 2 profiles (transcribed)
Below are direct transcriptions from local_lime_reports.json. LIME listed features as short rules/intervals — I parsed and summarized the main feature impacts.
Profile ID: 29 (LIME)
Predicted probability (default): 0.7056
True label: 1
Top LIME contributions (feature rule : weight):
0.00 < Savings_account_bonds_A61 <= 1.00 : +0.1583 (increases probability)
Duration_months > 0.44 : +0.0332 (increases)
Other_installment_plans_A141 <= 0.00 : -0.0325 (decreases)
Status_checking_account_A11 <= 0.00 : -0.0313 (decreases)
Status_checking_account_A14 <= 0.00 : +0.0297 (increases)
(remaining small contributions saved in local_lime_reports.json)
Comparison with SHAP: LIME and SHAP both indicate savings-account category and duration/checking-account rules as primary drivers; directions are consistent for the top contributors.
Profile ID: 74 (LIME)
Predicted probability (default): 0.6898
True label: 1
Top LIME contributions (feature rule : weight):
0.00 < Savings_account_bonds_A61 <= 1.00 : +0.1567
Duration_months > 0.44 : +0.0350
0.00 < Status_checking_account_A11 <= 1.00 : +0.0348
Other_installment_plans_A141 <= 0.00 : -0.0333
Status_checking_account_A14 <= 0.00 : +0.0286
(remaining small contributions saved in local_lime_reports.json)
Comparison with SHAP: LIME highlights similar rules (savings account bracket, duration, checking-account indicators) and matches SHAP on the direction of the strongest drivers.
(Full original LIME lists are available in local_lime_reports.json in the repo.)
8) Executive summary (1 paragraph)
A tuned LightGBM model trained on the UCI German Credit dataset achieved test AUC ≈ 0.786 and test F1 ≈ 0.494 after SMOTE oversampling and probability calibration. Global explainability identifies Status_checking_account, Credit_history, Credit_amount, Savings_account_bonds, and Purpose as the most influential predictors. Local SHAP force plots and LIME explanations explain individual predictions and align on top drivers. Business recommendations: require manual review for applicants with adverse checking-account indicators and large requested amounts, consider re-pricing for high-risk segments, and store SHAP/LIME transcripts in audit logs for regulatory review.
9) Reproducibility & run instructions
Open credit_risk_notebook_colab.ipynb in Google Colab.
Run all cells to install dependencies and execute the end-to-end pipeline.
Download credit_project_outputs.zip which contains the saved model, preprocessor, SHAP/LIME JSONs, and plots.
The JSONs included here (local_shap_reports.json, local_lime_reports.json) are the exact transcriptions used in the submission.
