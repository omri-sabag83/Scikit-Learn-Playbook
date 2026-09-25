# Scikit-Learn Playbook — Learning Program

**Goal:** working, defensible scikit-learn knowledge for analyst work: driver
analysis, honest model evaluation, and risk scoring. Enough to build a model,
evaluate it without fooling yourself, explain what drives it, and turn it into
a score someone can make decisions with. Not a deep dive into machine learning.

**Starting point:** a Product Analyst background (B2C marketplace, B2B security),
comfortable with SQL, Python, and statistics (hypothesis tests, confidence
intervals, regression basics, all covered in the A/B Testing Playbook). New
ground here: the scikit-learn modelling workflow, and the discipline of
*predictive* evaluation, where the question is "how well will this work on
data it hasn't seen?", not "is this coefficient significant?".

**Scope boundary:** clustering, heavy hyperparameter tuning, ensemble variants
and neural networks are out of scope (reserved for a possible Part II).

---

## Progress

`🟩⬜⬜⬜⬜⬜` **17% complete (1/6 modules)**

⬜ Not started · 🟨 In progress · 🟩 Completed

| # | Module | Status | Completed On |
|---|--------|--------|---------------|
| 1 | The Modelling Workflow: Splits, Pipelines, Cross-Validation & Leakage | 🟩 Completed | 2026-09-24 |
| 2 | Regression for Driver Analysis | ⬜ Not started | |
| 3 | One Tree Model, Interpreted with Permutation Importance | ⬜ Not started | |
| 4 | Evaluation Metrics under Class Imbalance | ⬜ Not started | |
| 5 | Calibration: Can You Trust the Probabilities? | ⬜ Not started | |
| 6 | The Risk Score: Building It and Deciding Whether to Trust It | ⬜ Not started | |

*The bar and the table get updated as modules are completed.*

---

## How this program works

Each module has:
- **Concepts**: what you need to understand, not just recite
- **Why it matters**: the real-world consequence of not knowing it, in analyst terms
- **Worked example**: a concrete illustration on real data
- **Resources**: optional, time-boxed, verified links; the in-module content is
  designed to be enough on its own
- **Exercise type**: delivered fully worked and explained, for review (the same
  convention as the A/B Testing Playbook)
- **Interview angle**: how it tends to come up in analyst interviews

**Standards applied in every notebook:** findings tagged **Observation** (what
the data shows), **Hypothesis** (a proposed explanation, not yet tested) or
**Conclusion** (what the evidence supports). Correlation vs. causation stated
explicitly for every driver. No evaluation ever touches data the model was fitted
on. Every printed statistic gets a plain-English sentence.

**Datasets** (both public, licensed CC BY 4.0, which allows reuse with attribution):
- **UCI Default of Credit Card Clients** (Yeh & Lien, 2009): Modules 2–6, the
  running dataset. 30,000 Taiwanese card holders (2005), 22.1% defaulted the
  following month. Credit scoring is the classic risk-score use case.
- **UCI Bank Marketing** (Moro, Laureano & Cortez, 2011): Module 1 only. 45,211
  telemarketing calls, 11.7% subscribed. Why the switch: it contains a real,
  documented leaky feature (`duration`), which teaches leakage better than a
  made-up example.

**The locked split** (Modules 2–6): the credit data is split once, with a fixed
seed, into 60% train / 20% validation / 20% test (see `common.py`). Modules 2–5
compare models only on cross-validation and the validation set. The test set is
used exactly once, in Module 6. Looking at the test set repeatedly while making
choices is the modelling equivalent of peeking at an A/B test.

---

## <u>Module 1 — The Modelling Workflow: Splits, Pipelines, Cross-Validation & Leakage</u>

**Concepts**
- Train/test split: why a model is judged on rows it never saw, and why the
  split is stratified (so both parts keep the same target rate)
- `Pipeline` and `ColumnTransformer`: preprocessing and model as one object, so
  every preprocessing step is fitted on training rows only
- Cross-validation (CV): k train/test splits instead of one, to average out
  the luck of a single split
- Data leakage, three kinds: (1) a feature that is only known *after* the
  outcome; (2) preprocessing fitted on all rows before splitting; (3) a random
  split when the real use is predicting the *future*

**Why it matters**
A leaky model looks excellent in the notebook and fails in production, and the
analyst who shipped it owns that failure. Every later module's evaluation is
only as honest as this workflow.

**Worked example**
Bank Marketing: predicting whether a call ends in a subscription. With the
call's `duration` as a feature, 5-fold CV gives an AUC (area under the ROC
curve, defined in the notebook) of about 0.91. Without it, about 0.77. But
duration is only known once the call has ended, i.e. after the outcome. The
0.91 model cannot exist at decision time.

**Resources**
- [Common pitfalls and recommended practices, §12.2 Data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
  (scikit-learn user guide, ~15 min, verified).
- [Leakage in Data Mining: Formulation, Detection, and Avoidance](https://dl.acm.org/doi/10.1145/2382577.2382579)
  (Kaufman, Rosset, Perlich & Stitelman, ACM TKDD 2012, verified). Skim
  sections 1–2 (~20 min): the standard definition of leakage, with real
  competition examples.

**Exercise type:** find-the-leak drills: given a set of candidate features and
preprocessing steps, classify each as safe or leaky and say why.

**Interview angle:** "Your model has 99% accuracy. What do you check first?" /
"How would you validate this model?"

---

## <u>Module 2 — Regression for Driver Analysis</u>

**Concepts**
- Linear regression (continuous target) and logistic regression (binary target)
  in scikit-learn, and reading their coefficients (for logistic: log-odds and
  odds ratios)
- scikit-learn's `LogisticRegression` applies an L2 penalty by default, which
  shrinks coefficients. That's fine for prediction but invalid for inference.
  statsmodels gives unpenalized estimates with p-values and confidence
  intervals (CIs)
- Standardized coefficients, to compare drivers measured in different units
- Multicollinearity (the six monthly repayment-status columns move together)
  and VIF (Variance Inflation Factor)
- A driver is an association inside this data, not a cause

**Why it matters**
"What drives default / churn / conversion?" is the most common modelling
question an analyst gets. Answering it with shrunken coefficients, or reading a
coefficient as a causal effect, is a common and costly mistake.

**Worked example**
Logistic regression of next-month default on repayment history, credit limit,
bill and payment amounts, and demographics: the same model fitted in
scikit-learn (penalized, for prediction) and in statsmodels (unpenalized, for
inference). Comparing the two sets of coefficients shows what the default
penalty changes, and the odds ratios translate the key driver into a
plain-English sentence.

**Resources**
- [Regression and Other Stories](https://avehtari.github.io/ROS-Examples/)
  (Gelman, Hill & Vehtari, 2020; free PDF linked from the book's homepage,
  verified). Ch. 10 "Linear regression with multiple predictors" and Ch. 13
  "Logistic regression", ~45 min each, optional.

**Exercise type:** interpret-the-output drills: turn a statsmodels summary into
driver statements, each tagged Observation/Hypothesis/Conclusion, with the
causal caveat stated.

**Interview angle:** "How would you find what drives churn?" / "How do you
interpret a logistic regression coefficient?"

---

## <u>Module 3 — One Tree Model, Interpreted with Permutation Importance</u>

**Concepts**
- A decision tree as a set of if/then splits; a Random Forest as many trees
  averaged (one model, used as-is, not a tour of ensembles)
- What a tree captures that a linear model misses: thresholds and interactions
- Impurity-based importance (built into the forest) vs. permutation importance
  (measured on held-out data), and why the first is biased
- Correlated features share or hide their importance, so correlated columns
  are permuted together as a group

**Why it matters**
"Which features matter most?" is the tree-model version of driver analysis. The
default importance plot is the one most often shown, and the one most often
wrong.

**Worked example**
A Random Forest on the credit data, compared with Module 2's logistic
regression on validation AUC. Then the forest's built-in importances vs.
permutation importances on the validation set, and grouped permutation of the
six correlated repayment-status columns.

**Resources**
- [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
  (scikit-learn user guide §5.2, ~15 min, verified). Covers exactly the two
  traps in this module: impurity bias and correlated features.

**Exercise type:** compare-and-reconcile: where the forest's drivers agree or
disagree with Module 2's regression, and why.

**Interview angle:** "How do you explain a black-box model to a stakeholder?" /
"What's wrong with feature importance?"

---

## <u>Module 4 — Evaluation Metrics under Class Imbalance</u>

**Concepts**
- Accuracy's trap: predicting "nobody defaults" is right 78% of the time here
- Confusion matrix, precision ("of those flagged, how many defaulted?"),
  recall ("of those who defaulted, how many did we flag?")
- ROC-AUC vs. PR-AUC (areas under the Receiver Operating Characteristic and
  Precision-Recall curves), and why a random model's PR-AUC equals the
  prevalence (the share of positives)
- Thresholds are a business decision, separate from the model
- `class_weight`: what it changes (the ranking of cases barely, the
  probabilities a lot, which Module 5 cares about)

**Why it matters**
Most analyst modelling targets are rare events (fraud, churn, default, a
security incident). Reporting a metric that looks good regardless of skill is
the evaluation version of leakage.

**Worked example**
A prevalence sweep: the same model scored on validation sets where defaulters
are progressively downsampled from 22% to about 2%. ROC-AUC barely moves, while
PR-AUC falls, because precision depends on how rare positives are. Same model,
same data: a controlled demonstration of what each metric does and doesn't see.

**Resources**
- [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0118432)
  (Saito & Rehmsmeier, PLOS ONE 2015, open access, verified). Introduction +
  figures, ~20 min.

**Exercise type:** metric-choice cases: for a given business problem and base
rate, which metric and threshold, and why.

**Interview angle:** "Why not use accuracy?" / "ROC-AUC or PR-AUC for fraud?"

---

## <u>Module 5 — Calibration: Can You Trust the Probabilities?</u>

**Concepts**
- Ranking (does the model put riskier cases higher?) vs. calibration (when it
  says 30%, do about 30% default?). They are different properties
- Reliability diagrams and the Brier score
- Why Random Forest probabilities, and anything trained with `class_weight`,
  come out miscalibrated
- `CalibratedClassifierCV`: sigmoid vs. isotonic, and why calibration must be
  fitted on rows the model was not trained on

**Why it matters**
A risk score is used as a probability: for expected-loss arithmetic, pricing,
and "flag everyone above 20%". If the probabilities are off, every decision
built on them is off, even when the ranking is excellent.

**Worked example**
Reliability diagrams for Module 2's logistic regression, Module 3's forest,
and a class-weighted model on the validation set, before and after calibration,
with the Brier score for each.

**Resources**
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
  (scikit-learn user guide §1.16, ~20 min, verified). Includes why forests
  produce a sigmoid-shaped reliability curve (Niculescu-Mizil & Caruana, 2005).

**Exercise type:** read-the-reliability-diagram drills: over- or
under-confident, where, and what it would cost.

**Interview angle:** "Your model says 30% risk. What does that mean, and how
would you check it?"

---

## <u>Module 6 — The Risk Score: Building It and Deciding Whether to Trust It</u>

*Gate: pause for Omri's go-ahead before starting this module.*

**Concepts**
- From probability to points: a multi-factor scorecard on the log-odds scale
  ("every 20 points doubles the odds of default"), as used in credit and
  insurance scoring
- Risk bands (A–E) and the checks a score must pass: default rates rise
  monotonically across bands, with confidence intervals; calibration holds on
  the untouched test set; performance is stable across subgroups
- Choosing a cut-off under explicit cost assumptions (cost of a missed default
  vs. cost of a wrongly declined client)
- Fairness caveat: sex, age and marital status are protected attributes in
  most lending regulation
- Bootstrap confidence intervals for the headline metrics

**Why it matters**
This is where the model becomes a decision tool. It's also the result that
needs the most scrutiny, so its headline claims are independently verified
(a fresh agent that sees only the claim and the data).

**Worked example**
The final scorecard evaluated once on the locked test set: band table, reliability
diagram, cost-based cut-off, and subgroup stability, each finding tagged.

**Resources** — verified when the module is built (after the gate).

**Exercise type:** a short score memo: what the score is, how well it works,
where it shouldn't be trusted.

**Interview angle:** "How would you build a risk score for X?" / "How would you
know if the score stopped working?"

---

## Open questions / decisions for as we go

- None yet.
