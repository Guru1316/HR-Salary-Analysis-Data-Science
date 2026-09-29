# %% [markdown]
# # HR — Is the Company Paying People Fairly? (DS_Day01_11)
# **Objective:** compensation-pattern analysis (not salary prediction). Linear Regression is used only as a supporting model.
#
# **Dataset:** `DS_Day01_11_HR_Salary_Inconsistent_Dataset.csv` — GitHub repo `suryap-0812/DS-Hackathon` (folder `Datasets/`).
# The repo README describes it as a hackathon dataset *intentionally built with inconsistencies* (i.e. synthetic/practice data),
# so findings describe **this dataset**, not a real company. Salary currency is not stated; values are annual and the locations are
# Indian cities, so INR is assumed. `Salary_Increment` is a percentage.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, joblib
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RS = 42
RAW, CLEAN = "../data/raw_dataset.csv", "../data/dataset.csv"
OUT, VIZ, MODEL = "../outputs/", "../outputs/visualizations/", "../model/linear_regression_model.pkl"
pd.set_option("display.width", 200, "display.max_columns", 30)
sns.set_theme(style="whitegrid", context="notebook")

# %% [markdown]
# ## Task 1 — Data cleaning & validation

# %%
raw = pd.read_csv(RAW)
print("Raw shape:", raw.shape)
print("\nDtypes:\n", raw.dtypes.to_string())
print("\nMissing values:\n", raw.isna().sum()[raw.isna().sum() > 0].to_string())
print("\nExact duplicate rows:", raw.duplicated().sum(), "| duplicated Employee_IDs:", raw.Employee_ID.duplicated().sum())
print("Employee_ID format invalid:", (~raw.Employee_ID.astype(str).str.match(r"^EMP_\d{3}$")).sum())
print("Experience < 0:", (raw.Experience < 0).sum(), "| Experience > 40:", (raw.Experience > 40).sum())
print("Salary <= 0:", (raw.Salary <= 0).sum())
print("Performance_Score outside 1-5:", (~raw.Performance_Score.between(1, 5) & raw.Performance_Score.notna()).sum())
print("Salary_Increment outside 0-15%:", (~raw.Salary_Increment.between(0, 15)).sum())
for c in ["Designation", "Education", "Location"]:
    print(f"\n{c} raw values:", sorted(raw[c].dropna().unique().tolist(), key=str))

# %%
log = []
df = raw.copy()

# 1) categorical consistency: strip/case/aliases; 'Unknown' is not a real value -> missing
DES = {"junior developer": "Junior Developer", "software engineer": "Software Engineer",
       "senior software engineer": "Senior Software Engineer", "hr executive": "HR Executive",
       "tech lead": "Tech Lead", "project manager": "Project Manager", "data analyst": "Data Analyst", "intern": "Intern"}
EDU = {"bachelor's": "Bachelor's", "bachelors": "Bachelor's", "master's": "Master's", "masters": "Master's",
       "mba": "MBA", "phd": "PhD", "diploma": "Diploma"}
LOC = {k: k.title() for k in ["bangalore", "hyderabad", "mumbai", "chennai", "pune", "delhi", "remote"]}
changed = 0
for col, mp in [("Designation", DES), ("Education", EDU), ("Location", LOC)]:
    key = df[col].astype("string").str.strip().str.lower()
    changed += int((df[col].notna() & (df[col] != key.map(mp))).sum())
    df[col] = key.map(mp)   # anything not in mapping (e.g. 'unknown') becomes NaN
log.append(("Categorical values standardised (case/whitespace/aliases) or 'Unknown' -> missing", changed, "cell values"))

# 2) exact duplicates
n = df.duplicated().sum(); df = df.drop_duplicates().reset_index(drop=True)
log.append(("Exact duplicate rows removed", int(n), "rows removed"))
assert df.Employee_ID.is_unique

# 3) invalid numeric values -> set to missing (never imputed)
inv_perf = ~df.Performance_Score.between(1, 5) & df.Performance_Score.notna()
inv_inc = ~df.Salary_Increment.between(0, 15)
df.loc[inv_perf, "Performance_Score"] = np.nan
df.loc[inv_inc, "Salary_Increment"] = np.nan
log.append(("Performance_Score outside 1-5 set to missing (row kept)", int(inv_perf.sum()), "values"))
log.append(("Salary_Increment outside 0-15% (-3, 25) set to missing (row kept)", int(inv_inc.sum()), "values"))

# 4) rows lacking a valid core field cannot be analysed for pay patterns -> removed (reported, not imputed)
core_bad = (df.Experience < 0) | (df.Experience > 40) | (df.Salary <= 0) | df[["Designation", "Education", "Location", "Salary"]].isna().any(axis=1)
reasons = {"Experience invalid (<0 or >40)": ((df.Experience < 0) | (df.Experience > 40)).sum(),
           "Salary missing or <=0": (df.Salary.isna() | (df.Salary <= 0)).sum(),
           "Designation missing/Unknown": df.Designation.isna().sum(),
           "Education missing/Unknown": df.Education.isna().sum(),
           "Location missing/Unknown": df.Location.isna().sum()}
for k, v in reasons.items(): log.append((f"  reason (may overlap): {k}", int(v), "rows"))
log.append(("Rows removed for invalid/missing core field", int(core_bad.sum()), "rows removed"))
df = df[~core_bad].reset_index(drop=True)
log.append(("Rows retained (final cleaned dataset)", len(df), f"of {len(raw)} raw ({len(df)/len(raw):.1%})"))
print(pd.DataFrame(log, columns=["Step", "Count", "Unit"]).to_string(index=False))

print("\nRemaining missing:", df.isna().sum()[df.isna().sum() > 0].to_dict(), "(Performance_Score/Increment kept as missing; excluded only where needed)")
# basic outlier inspection (IQR fences) - flagged, retained if plausible
for c in ["Salary", "Experience", "Salary_Increment"]:
    q1, q3 = df[c].quantile([.25, .75]); f = 1.5 * (q3 - q1)
    print(f"IQR outliers in {c}: {((df[c] < q1 - f) | (df[c] > q3 + f)).sum()}")
print("Note: Junior Developer with unusually high experience retained (valid value, examined in peer analysis):")
print(df[(df.Designation == "Junior Developer") & (df.Experience > 8)][["Employee_ID", "Experience", "Salary"]].to_string(index=False))

pd.DataFrame(log, columns=["Step", "Count", "Unit"]).to_csv(OUT + "cleaning_log.csv", index=False)
df.to_csv(CLEAN, index=False)
print("\nSaved cleaned dataset:", df.shape)

# %% [markdown]
# ## Task 2 — Descriptive statistics

# %%
bands = [-1, 1, 3, 6, 10, 100]; blabels = ["0-1", "2-3", "4-6", "7-10", "11+"]
df["Exp_Band"] = pd.cut(df.Experience, bins=bands, labels=blabels)
num = ["Salary", "Experience", "Salary_Increment", "Performance_Score"]
desc = df[num].agg(["count", "mean", "median", "min", "max", "std"]).T.round(2)
print(desc.to_string()); desc.to_csv(OUT + "descriptive_overall.csv")
for c in ["Designation", "Education", "Location"]:
    print(f"\n{c} counts:", df[c].value_counts().to_dict())

def grp(by):
    g = df.groupby(by, observed=True).agg(n=("Salary", "size"), mean_salary=("Salary", "mean"), median_salary=("Salary", "median"),
        min_salary=("Salary", "min"), max_salary=("Salary", "max"), std_salary=("Salary", "std"),
        avg_experience=("Experience", "mean"), avg_increment=("Salary_Increment", "mean"))
    return g.round(1)
for by in ["Education", "Exp_Band"]:
    t = grp(by); print(f"\nSalary stats by {by}:\n", t.to_string()); t.to_csv(OUT + f"stats_by_{by.lower()}.csv")

# %% [markdown]
# ## Task 3 — Salary differences across job roles

# %%
role = grp("Designation").sort_values("median_salary", ascending=False)
print(role.to_string()); role.to_csv(OUT + "salary_by_designation.csv")
# Context: raw role gaps are confounded by experience -> compare roles inside the overlapping 2-6 year range
ov = df[df.Experience.between(2, 6)]
ovt = ov.groupby("Designation").agg(n=("Salary", "size"), avg_exp=("Experience", "mean"), median_salary=("Salary", "median")).round(1)
print("\nRoles compared at overlapping experience (2-6 yrs):\n", ovt[ovt.n >= 5].sort_values("median_salary", ascending=False).to_string())
print("\nExperience range by role:\n", df.groupby("Designation").Experience.agg(["min", "median", "max"]).to_string())
ovt.to_csv(OUT + "salary_by_designation_exp_2_6.csv")

# %% [markdown]
# ## Task 4 — Salary growth with experience

# %%
r, p = stats.pearsonr(df.Experience, df.Salary); rho, prho = stats.spearmanr(df.Experience, df.Salary)
print(f"Experience vs Salary: Pearson r={r:.3f} (p={p:.2g}), Spearman rho={rho:.3f}  (n={len(df)})")
eb = grp("Exp_Band")[["n", "mean_salary", "median_salary", "min_salary", "max_salary", "std_salary"]]
print("\nSalary by experience band:\n", eb.to_string()); eb.to_csv(OUT + "salary_by_experience_band.csv")
# Within-role association (removes the role/experience overlap)
rows = []
for d, g in df.groupby("Designation"):
    if len(g) >= 20 and g.Experience.nunique() > 2:
        rr, pp = stats.spearmanr(g.Experience, g.Salary)
        slope = np.polyfit(g.Experience, g.Salary, 1)[0]
        rows.append((d, len(g), round(rr, 3), round(pp, 3), round(slope)))
within = pd.DataFrame(rows, columns=["Designation", "n", "spearman_rho", "p_value", "slope_salary_per_year"])
print("\nWithin-role association (roles with n>=20):\n", within.to_string(index=False)); within.to_csv(OUT + "experience_salary_within_role.csv", index=False)

# %% [markdown]
# ## Task 5 — Peer compensation comparison
# **Peer definition (transparent):** for each employee, peers = *other* employees with the **same Designation** and Experience within **±2 years**.
# If fewer than 5 such peers exist, fall back to *same Designation only* (labelled). If the Designation has <5 other employees, the employee is not assessed.
# Location/Education are not in the peer key because groups would fall below 5 people; their association with salary is examined in Tasks 6 and 11.
# **Rule:** |Salary difference %| ≥ 25% → "Substantial Peer Deviation / Compensation Review Candidate" (a screening threshold, not a fairness verdict).

# %%
MIN_PEERS, THRESH = 5, 25.0
def peers(row):
    same = df[(df.Designation == row.Designation) & (df.Employee_ID != row.Employee_ID)]
    win = same[(same.Experience - row.Experience).abs() <= 2]
    if len(win) >= MIN_PEERS: return win.Salary.median(), len(win), "Designation +/-2 yrs experience"
    if len(same) >= MIN_PEERS: return same.Salary.median(), len(same), "Designation only (fallback)"
    return np.nan, len(same), "Not assessed (too few peers)"
res = df.apply(peers, axis=1, result_type="expand"); res.columns = ["Peer_Median_Salary", "Peer_N", "Peer_Basis"]
pe = pd.concat([df, res], axis=1)
pe["Salary_Diff"] = pe.Salary - pe.Peer_Median_Salary
pe["Salary_Diff_Pct"] = (pe.Salary_Diff / pe.Peer_Median_Salary * 100).round(1)
pe["Review_Flag"] = np.where(pe.Salary_Diff_Pct >= THRESH, "Review Candidate - above peer median",
                     np.where(pe.Salary_Diff_Pct <= -THRESH, "Review Candidate - below peer median", "Within range"))
pe.loc[pe.Peer_Median_Salary.isna(), "Review_Flag"] = "Not assessed"
print("Peer basis:", pe.Peer_Basis.value_counts().to_dict())
print("\nSalary_Diff_Pct distribution (assessed employees):\n", pe.Salary_Diff_Pct.describe().round(1).to_string())
print("\nReview flags:", pe.Review_Flag.value_counts().to_dict())
for t in (15, 20, 25, 30, 40):   # sensitivity of the threshold
    print(f"  threshold ±{t}%: {(pe.Salary_Diff_Pct.abs() >= t).sum()} employees")
cols = ["Employee_ID", "Designation", "Experience", "Education", "Location", "Performance_Score", "Salary", "Peer_Median_Salary", "Peer_N", "Peer_Basis", "Salary_Diff", "Salary_Diff_Pct", "Review_Flag"]
cand = pe[pe.Review_Flag.str.startswith("Review")].sort_values("Salary_Diff_Pct", key=abs, ascending=False)[cols]
print("\nTop 12 largest deviations:\n", cand.head(12).to_string(index=False))
print("\nCandidates by designation:\n", pd.crosstab(cand.Designation, cand.Review_Flag).to_string())
ass = pe[pe.Review_Flag != "Not assessed"]
rate = ass.groupby("Designation").agg(n=("Salary", "size"), median_salary=("Salary", "median"),
    flagged=("Review_Flag", lambda x: int(x.str.startswith("Review").sum())),
    iqr_of_diff_pct=("Salary_Diff_Pct", lambda x: x.quantile(.75) - x.quantile(.25)))
rate["flag_rate_%"] = (rate.flagged / rate.n * 100).round(1); rate["iqr_of_diff_pct"] = rate.iqr_of_diff_pct.round(1)
print("\nFlag rate and within-role dispersion of peer deviation by designation:\n", rate.to_string()); rate.to_csv(OUT + "peer_flag_rate_by_designation.csv")
print("Fallback-basis candidates:", (cand.Peer_Basis != "Designation +/-2 yrs experience").sum())
pe[cols].to_csv(OUT + "peer_comparison_all_employees.csv", index=False)
cand.to_csv(OUT + "compensation_review_candidates.csv", index=False)

# %% [markdown]
# ## Task 6 — Salary structures across locations

# %%
loc = grp("Location").sort_values("median_salary", ascending=False)
senior = ["Tech Lead", "Project Manager", "Senior Software Engineer"]
loc["senior_role_share_%"] = (df.assign(s=df.Designation.isin(senior)).groupby("Location").s.mean() * 100).round(1)
loc["median_peer_dev_%"] = pe.groupby("Location").Salary_Diff_Pct.median().round(1)
loc["review_candidates"] = cand.groupby("Location").size().reindex(loc.index).fillna(0).astype(int)
print(loc.to_string()); loc.to_csv(OUT + "salary_by_location.csv")
print("\nReview-candidate direction by location:\n", pd.crosstab(cand.Location, cand.Review_Flag.str.replace("Review Candidate - ", "")).to_string())
print("\nRaw median-salary spread across major locations (n>=20):", end=" ")
big = loc[loc.n >= 20]; print(f"{big.median_salary.min():,.0f} to {big.median_salary.max():,.0f}")
print("Median salary by Location x Designation (n>=3 cells):")
lt = df.groupby(["Designation", "Location"]).Salary.agg(["size", "median"]); lt = lt[lt["size"] >= 3]["median"].unstack().round(0)
print(lt.to_string())
kw = stats.kruskal(*[g.Salary.values for _, g in df[df.Location.isin(big.index)].groupby("Location")])
print(f"\nKruskal-Wallis on raw salary across major locations: H={kw.statistic:.2f}, p={kw.pvalue:.3f}")
kw2 = stats.kruskal(*[g.Salary_Diff_Pct.dropna().values for _, g in pe[pe.Location.isin(big.index)].groupby("Location")])
print(f"Kruskal-Wallis on peer-adjusted deviation % across major locations: H={kw2.statistic:.2f}, p={kw2.pvalue:.3f}")

# %% [markdown]
# ## Task 7 — Performance score vs salary increment

# %%
pf = df.dropna(subset=["Performance_Score", "Salary_Increment"])
r7, p7 = stats.pearsonr(pf.Performance_Score, pf.Salary_Increment); rho7, prho7 = stats.spearmanr(pf.Performance_Score, pf.Salary_Increment)
print(f"Performance vs Increment (n={len(pf)}): Pearson r={r7:.3f} (p={p7:.2g}), Spearman rho={rho7:.3f} (p={prho7:.2g})")
pt = pf.groupby("Performance_Score").Salary_Increment.agg(n="size", mean="mean", median="median", std="std", min="min", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), max="max").round(2)
print(pt.to_string()); pt.to_csv(OUT + "increment_by_performance.csv")
print("Performance vs Salary (for context): Pearson r=%.3f" % stats.pearsonr(pf.Performance_Score, pf.Salary)[0])

# %% [markdown]
# ## Task 8 — Four main visualizations

# %%
d = df.assign(Salary_L=df.Salary / 1e5)   # lakhs, plotting only
order = d.groupby("Designation").Salary_L.median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(data=d, x="Designation", y="Salary_L", order=order, palette="Blues", showfliers=False, ax=ax)
sns.stripplot(data=d, x="Designation", y="Salary_L", order=order, color="k", size=2.5, alpha=.5, ax=ax)
ax.set_xticklabels([f"{t}\n(n={int((d.Designation == t).sum())})" for t in order], rotation=0, fontsize=8)
ax.set_title("Viz 1: Salary Distribution by Designation"); ax.set_ylabel("Annual salary (lakhs; INR assumed)"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(VIZ + "1_salary_by_designation.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=d, x="Experience", y="Salary_L", hue="Designation", hue_order=order, palette="tab10", s=35, alpha=.8, ax=ax)
sns.regplot(data=d, x="Experience", y="Salary_L", scatter=False, color="black", line_kws={"lw": 2, "ls": "--"}, ax=ax)
ax.set_title(f"Viz 2: Experience vs Salary (overall linear trend, Pearson r = {r:.2f})")
ax.set_xlabel("Experience (years)"); ax.set_ylabel("Annual salary (lakhs; INR assumed)"); ax.legend(fontsize=8, title="Designation")
plt.tight_layout(); plt.savefig(VIZ + "2_experience_vs_salary.png", dpi=150); plt.close()

lorder = d.groupby("Location").Salary_L.median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=d, x="Location", y="Salary_L", order=lorder, palette="Greens", showfliers=False, ax=ax)
sns.stripplot(data=d, x="Location", y="Salary_L", order=lorder, color="k", size=2.5, alpha=.5, ax=ax)
ax.set_xticklabels([f"{t}\n(n={int((d.Location == t).sum())})" for t in lorder], fontsize=9)
ax.set_title("Viz 3: Salary Comparison Across Locations (not adjusted for role/experience mix)"); ax.set_ylabel("Annual salary (lakhs; INR assumed)"); ax.set_xlabel("")
plt.tight_layout(); plt.savefig(VIZ + "3_salary_by_location.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=pf, x="Performance_Score", y="Salary_Increment", palette="Oranges", showfliers=False, ax=ax)
sns.stripplot(data=pf, x="Performance_Score", y="Salary_Increment", color="k", size=2.5, alpha=.5, ax=ax)
ax.set_title(f"Viz 4: Performance Score vs Salary Increment (Pearson r = {r7:.2f}, n = {len(pf)})")
ax.set_xlabel("Performance score"); ax.set_ylabel("Salary increment (%)")
plt.tight_layout(); plt.savefig(VIZ + "4_performance_vs_increment.png", dpi=150); plt.close()
print("Saved 4 visualizations")

# %% [markdown]
# ## Tasks 9–10 — Linear Regression (supporting analytical model) and evaluation

# %%
feat_cat = ["Designation", "Education", "Location"]; feat_num = ["Experience", "Performance_Score"]
md = df.dropna(subset=["Performance_Score"]).copy()
print(f"Modelling rows: {len(md)} (rows with missing Performance_Score excluded, not imputed)")
X, y = md[feat_cat + feat_num], md.Salary
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
# reference categories = most common, so coefficients read as "difference vs reference"
ref = [["Software Engineer"], ["Bachelor's"], ["Bangalore"]]
ohe = OneHotEncoder(drop=[r[0] for r in ref], handle_unknown="ignore", sparse_output=False)
pre = ColumnTransformer([("cat", ohe, feat_cat), ("num", "passthrough", feat_num)])
model = Pipeline([("prep", pre), ("lr", LinearRegression())]).fit(Xtr, ytr)
pred = model.predict(Xte)
mae, rmse, r2 = mean_absolute_error(yte, pred), np.sqrt(mean_squared_error(yte, pred)), r2_score(yte, pred)
cv = cross_val_score(Pipeline([("prep", pre), ("lr", LinearRegression())]), X, y, cv=5, scoring="r2")
print(f"Train rows={len(Xtr)}, Test rows={len(Xte)}, random_state={RS}")
print(f"TEST  MAE = {mae:,.0f} | RMSE = {rmse:,.0f} | R2 = {r2:.3f}")
print(f"Train R2 = {model.score(Xtr, ytr):.3f} | 5-fold CV R2 = {cv.mean():.3f} (±{cv.std():.3f})  [stability check, same pipeline]")
print(f"Context: salary mean={y.mean():,.0f}, std={y.std():,.0f}; MAE is {mae/y.mean():.1%} of mean salary")
# Management hypothesis check: same algorithm, restricted features (Designation + Experience only)
pre2 = ColumnTransformer([("cat", OneHotEncoder(drop=["Software Engineer"], handle_unknown="ignore", sparse_output=False), ["Designation"]), ("num", "passthrough", ["Experience"])])
m2 = Pipeline([("prep", pre2), ("lr", LinearRegression())]).fit(Xtr, ytr); p2 = m2.predict(Xte)
cv2 = cross_val_score(Pipeline([("prep", pre2), ("lr", LinearRegression())]), X, y, cv=5, scoring="r2")
print(f"Designation+Experience only: TEST MAE={mean_absolute_error(yte, p2):,.0f} RMSE={np.sqrt(mean_squared_error(yte, p2)):,.0f} R2={r2_score(yte, p2):.3f} | 5-fold CV R2={cv2.mean():.3f}")
print(f"Adding Education+Location+Performance: CV R2 gain = {cv.mean() - cv2.mean():+.3f}")
big_err = pd.Series(np.abs(yte - pred)).sort_values(ascending=False).head(3)
print("Largest test errors (abs):", [f"{md.loc[i, 'Employee_ID']} ({md.loc[i, 'Designation']}, exp {md.loc[i, 'Experience']}): {v:,.0f}" for i, v in big_err.items()])
joblib.dump(model, MODEL); print("Saved model:", MODEL)

# %% [markdown]
# ## Task 11 — Model interpretation (association only, not causal)

# %%
names = model.named_steps["prep"].get_feature_names_out()
coef = pd.DataFrame({"feature": [n.split("__")[1] for n in names], "coefficient": model.named_steps["lr"].coef_.round(0)})
counts = {f"{c}_{v}": n for c in feat_cat for v, n in md[c].value_counts().items()}
coef["n_in_data"] = coef.feature.map(counts)
print("Intercept:", round(model.named_steps["lr"].intercept_))
print("References: Designation=Software Engineer, Education=Bachelor's, Location=Bangalore\n")
print(coef.to_string(index=False)); coef.to_csv(OUT + "linear_regression_coefficients.csv", index=False)
pd.DataFrame({"metric": ["MAE", "RMSE", "R2_test", "R2_5fold_cv_mean", "R2_test_designation_experience_only", "R2_cv_designation_experience_only"], "value": [round(mae), round(rmse), round(r2, 4), round(cv.mean(), 4), round(r2_score(yte, p2), 4), round(cv2.mean(), 4)]}).to_csv(OUT + "model_evaluation.csv", index=False)
