# Aura — AI Big Five Personality Predictor

A 20-question personality quiz that predicts your Big Five (OCEAN) trait
scores using a regression model trained on real survey data, visualizes your
profile on an animated radar chart, maps you onto the wider population with
PCA, and generates a shareable result card.

**Live demo:** open `personality-predictor.html` in any browser, or deploy it
as a static site (see [Deploying](#deploying) below).

---

## What's in this repo

| File | What it is |
|---|---|
| `index.html` | The whole app. Single file, no build step, no backend. |
| `train_model.py` | The offline training script. Run this if you want to retrain the model yourself. |

Everything the page needs (model weights, PCA components, a population
sample) is baked into `personality-predictor.html` as a JSON blob near the
top of the `<script>` tag. **The page does inference only — it never trains
anything.** Training happens once, offline, in Python.

---

## How it works

### 1. Data
[`tunguz/big-five-personality-test`](https://www.kaggle.com/datasets/tunguz/big-five-personality-test)
on Kaggle — real responses to the IPIP-50 "Big-Five Factor Markers" test,
collected online by Open Psychometrics (2016–2018). This project uses a
38,000-row slice of the full ~1,000,000-row dataset (the full file is large;
the slice is plenty for this).

After cleaning out invalid/missing rows: **32,517 real respondents.**

### 2. Reverse-key detection
The IPIP-50 mixes straight items ("I have a rich vocabulary") with reverse
items ("I have difficulty understanding abstract ideas"). Rather than typing
in a scoring key from memory, the script detects reverse-keyed items
**from the data itself**: for each trait's 10 items, it takes the first
eigenvector of their correlation matrix — items that load negatively are the
reverse-keyed ones. This is more reliable than a memorized key, and it
actually caught a wrong assumption during development (one openness item
turned out to be straight, not reverse, contrary to what's commonly cited).

### 3. Short-form scoring
For each trait, the script:
- Computes the validated **full 10-item trait score** (0–100) using the
  detected key — this is the "ground truth" target.
- Picks the **4 most representative items** (highest item-rest correlation)
  to use as the quiz questions.
- Fits a **Ridge regression** (`scikit-learn`, L2-regularized linear
  regression) that predicts the full 10-item score from just those 4 raw
  answers, using an 80/20 train/test split.
- Also fits a **Random Forest** on the same task as a comparison baseline.

Ridge slightly outperformed Random Forest on every trait, and is far more
portable (5 numbers per trait vs. a forest of trees), so Ridge is what ships.

**Held-out test performance** (predicting a 10-item scale from 4 items):

| Trait | Test R² | RMSE |
|---|---|---|
| Openness | 0.83 | 6.35 |
| Conscientiousness | 0.84 | 7.26 |
| Extraversion | 0.89 | 7.39 |
| Agreeableness | 0.84 | 7.10 |
| Neuroticism | 0.87 | 7.51 |

### 4. PCA
`PCA(n_components=2)` from scikit-learn, fit on the 5 trait scores across
the full cleaned population. The two components together explain ~58% of
the variance between people. A 900-row random sample of real respondents
(just their 5 trait scores + 2D projection — no identifying info of any
kind) is shipped with the page to draw the "population cloud" scatter plot.

### 5. What ships in the HTML
A single JSON object, `MODEL_DATA`, containing:
- Per trait: which 4 items were chosen, their reverse-key flags, and the 5
  Ridge weights (4 coefficients + intercept)
- R² / RMSE per trait, for the in-app "About the model" panel
- PCA mean vector + the 2 component vectors
- The 900-row population sample (scores + projections)
- Population mean scores (for the "compare to average" radar overlay)

The frontend then just does arithmetic: dot products for trait prediction,
dot products for PCA projection. No training, no backend, no network calls
after the page loads.

---

## Notes & limitations

- This is a **short-form predictor**, not the real test: it estimates the
  full 10-item validated score from 4 items, with the error margins shown
  above (RMSE ~6–7.5 points on a 0–100 scale). It's a fun, genuinely
  data-driven estimate — not a clinical or research-grade assessment.
- The Big Five model itself, like any personality framework, is a
  simplification. Scores are descriptive snapshots, not fixed labels.
- The PCA axes are population-relative and will shift slightly if you
  retrain on different data — they're a lens for "where you sit relative to
  this sample," not a fixed, universally meaningful coordinate system.
