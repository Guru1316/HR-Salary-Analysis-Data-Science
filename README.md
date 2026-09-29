# HR — Is the Company Paying People Fairly? (DS_Day01_11)

Compensation-pattern analysis (not salary prediction). Linear Regression is used only as a supporting analytical model.

**Dataset:** `DS_Day01_11_HR_Salary_Inconsistent_Dataset.csv`, GitHub repo `suryap-0812/DS-Hackathon` (folder `Datasets/`). 270 raw rows, 8 columns that match the required fields exactly (no mapping needed). The repo describes it as intentionally inconsistent hackathon/practice data (synthetic), so results describe this dataset, not a real company. Currency is not stated; INR is assumed.

**Run:** `pip install -r requirements.txt`, then execute `notebook/hr_salary_analysis.ipynb` (paths are relative to `notebook/`; raw data in `data/raw_dataset.csv`).

## Structure
- `data/raw_dataset.csv`, `data/dataset.csv` (cleaned, 234 rows)
- `notebook/hr_salary_analysis.ipynb` (full analysis, executed) and `notebook/source.py`, `build.py`, `insights.md` (notebook sources)
- `outputs/*.csv` (cleaning log, stats tables, peer comparison, review candidates, coefficients, model evaluation)
- `outputs/visualizations/` (4 main charts)
- `model/linear_regression_model.pkl` (OneHotEncoder + LinearRegression pipeline, fit on the 80% training split, random_state=42)

## Method notes
- Cleaning: 270 -> 234 rows (10 exact duplicates, 26 rows with an invalid/missing core field). Invalid Performance_Score (2) and Salary_Increment (2) values were set to missing, not imputed.
- Peer group: same Designation, Experience within ±2 years (min 5 peers, else same Designation only, labelled). Flag = |difference from peer median| >= 25%.
- Flags are "Compensation Review Candidates", not findings of under/over-payment, bias or discrimination.

## Results
See the last section of the notebook (Task 12, 7 insights) for evidence-backed findings and the final answer to the guiding question.
