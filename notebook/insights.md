## Task 12 — Key insights (all figures from this notebook's outputs; salary in INR assumed, 1 lakh = 100,000)

**Insight 1 — Role pay tiers are wide, but roles sit in different experience ranges.**
Finding: Median salary runs from 6.1 lakh (Junior Developer) and 6.6 lakh (HR Executive) up to 25.0 lakh (Tech Lead).
Evidence: Tech Leads have 8–17 yrs experience vs 0–2 yrs for Data Analysts/HR Executives. In the overlapping 2–6 yr range, Project Managers (n=21, avg 4.8 yrs) have a median of 17.2 lakh vs 10.1 lakh for Software Engineers (n=66, avg 3.8 yrs).
Interpretation: Raw role gaps mix seniority with role; a ~1-year experience difference does not explain the PM–Software Engineer gap, so role structure itself is associated with pay.

**Insight 2 — Salary is strongly associated with experience overall, but much less uniformly within roles.**
Finding: Salary shows a strong positive association with experience (Pearson r = 0.90, Spearman 0.88, n = 234); band medians rise from 6.5 lakh (0–1 yrs) to 24.8 lakh (11+ yrs).
Evidence: Within-role Spearman ranges from 0.23 (Junior Developer, p = 0.147) to 0.76 (Senior Software Engineer); Software Engineers gain ~32k per extra year vs ~100k for Data Analysts.
Interpretation: The overall correlation is inflated because senior titles coincide with higher experience; experience does not progress at one common rate within each role.

**Insight 3 — 43 of 233 assessed employees (18.5%) are Substantial Peer Deviation / Compensation Review Candidates (±25% vs peer median): 24 above, 19 below.**
Finding: Deviations are concentrated in Junior Developers: 21 of 42 flagged (50%), vs 0–24% in every other role.
Evidence: Junior Developer peer-deviation IQR is 50.8 points vs 12.5–32.6 for other roles. Largest cases: EMP_213 (Junior Developer, 16 yrs, +356%, compared to all Junior Developers as no similarly experienced peers exist), EMP_214 (+98.7%), EMP_236 (+90.5%); lowest EMP_069 (−53.7%), EMP_025 (−53.6%). Count is threshold-sensitive: 76 / 56 / 43 / 30 / 13 employees at ±15 / 20 / 25 / 30 / 40%.
Interpretation: These are screening flags, not findings of under/over-payment; Junior Developer pay appears the least consistent and is the first place to look.

**Insight 4 — Location differences persist after accounting for role and experience.**
Finding: Raw median salary is ~10.9–11.1 lakh in Delhi/Bangalore/Mumbai vs ~9.1–9.3 lakh in Hyderabad/Pune/Chennai, but raw figures are affected by mix (senior-role share: Delhi 47%, Bangalore 40% vs Hyderabad 24%).
Evidence: Median peer-adjusted deviation: Bangalore +9.8%, Mumbai +6.6%, Hyderabad −2.1%, Delhi −3.2%, Pune −3.5%, Chennai −7.8% (Kruskal-Wallis p < 0.001). Review candidates above peers: Bangalore 11 vs 2 below; Hyderabad 3 above vs 8 below. Regression vs Bangalore: Chennai −2.35 lakh, Pune −1.50, Hyderabad −1.42, Delhi −0.97, Mumbai −0.17.
Interpretation: Delhi looks high in raw terms but not after peer adjustment (workforce mix). Adjusted gaps remain for Bangalore/Mumbai vs Chennai/Hyderabad; whether they reflect an intended geographic pay policy cannot be determined from this data.

**Insight 5 — Performance is closely associated with salary increments, but not with salary levels.**
Finding: Higher performance scores are associated with higher increments in the observed dataset (r = 0.85, n = 227).
Evidence: Mean increment by score 1→5: 2.05%, 4.51%, 6.25%, 7.69%, 9.85%. Performance vs salary level: r = −0.09. A few mismatches exist, e.g. EMP_239 (Tech Lead, score 4, 1.0%), EMP_003 (score 2, 7.4%), EMP_069 (score 3, 10.0%).
Interpretation: Increments follow ratings, so pay-for-performance operates through increments; current salary levels are explained by role/experience rather than performance.

**Insight 6 — Designation + Experience already explain most salary variation (supports management's view), with a modest extra contribution from other factors.**
Finding: A Linear Regression using only Designation + Experience reaches 5-fold CV R² = 0.889; adding Education, Location and Performance raises it to 0.911 (+0.022).
Evidence: Full model, held-out test (n=46): MAE 138,596; RMSE 268,817; R² 0.801 (restricted model on the same split: R² 0.814). Largest test errors are EMP_066 and EMP_025, both also peer-flagged.
Interpretation: The model is only a description of association in this dataset; it is not a statement of what anyone should earn.

**Insight 7 — Fitted-model associations (conditional on other variables; not causal).**
Finding: Each extra year of experience is associated with ~+69k salary; roles vs Software Engineer: Tech Lead +8.3 lakh, Project Manager +6.2 lakh, Senior SE +3.8 lakh, Junior Developer −2.4 lakh, HR Executive −2.3 lakh.
Evidence: Education vs Bachelor's: Master's +1.1 lakh, MBA +0.9 lakh, PhD +2.9 lakh (n = 7 only); Performance +53k per score point. Categories with n = 1 (Intern, Diploma, Remote) are not interpretable.
Interpretation: Education and performance associations are generally smaller than role associations; the PhD estimate is unreliable due to small n.

**Answer to "Is the company paying people fairly?"** In this dataset pay is largely structured by designation and experience, but 43 individual cases, Junior Developer pay dispersion and location gaps merit review. This dataset cannot establish fairness, bias or discrimination.
