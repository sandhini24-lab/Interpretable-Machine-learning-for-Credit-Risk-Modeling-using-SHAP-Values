# Executive summary - German Credit Risk (UCI)

**Model:** LightGBM (tuned) with SHAP explanations.

**Performance (test):**
- AUC: 0.786
- F1 (at threshold 0.43): 0.614

**Top drivers (global SHAP):**
- Status_checking_account_A14: mean |SHAP| = 0.264773
- Savings_account_bonds_A61: mean |SHAP| = 0.122558
- Duration_months: mean |SHAP| = 0.112240
- Status_checking_account_A11: mean |SHAP| = 0.096179
- Installment_rate: mean |SHAP| = 0.069781
- Credit_history_A34: mean |SHAP| = 0.057980
- Credit_amount: mean |SHAP| = 0.050373

**Top underwriting recommendations:**
- Strengthen review for applicants with high values in the top SHAP drivers.
- Consider re-pricing/raising interest or decreasing loan amount for high-risk segments.
- Add manual review for borderline profiles identified by local SHAP force plots.
