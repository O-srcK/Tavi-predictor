# Tavi_predictor — post-TAVI complication risk predictor

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Bloc05kadeOscar/tavi-predictor)

**▶ Try it live:** [https://huggingface.co/spaces/Bloc05kadeOscar/tavi-predictor](https://huggingface.co/spaces/Bloc05kadeOscar/tavi-predictor)


A machine-learning model that estimates the risk of complications after **TAVI** (Transcatheter Aortic Valve Implantation) from a small set of clinical measurements, served through a simple **Flask** web app. Built as the machine-learning component of an accepted Master's thesis.

## What it does

Given two clinical inputs, the model returns a complication-risk probability (0–100%). It was deliberately kept to **2 features** so it's practical to use at the bedside while still giving useful discrimination.

- **Model:** Logistic Regression (L2-regularized)
- **Features:** `FCC` and `Masa VS/SC (g/m²)` (ventricular mass normalized by body surface area)
- **Data:** 179 patients, 18 with complications (~10% prevalence)
- **Performance:** AUC ≈ 0.83; recall tunable via threshold presets (from maximum-safety to maximum-accuracy)

## Repository layout

- **`try 12/`** — the final, production-ready version: training pipeline, saved model (`final_model.pkl`), evaluation scripts, and the Flask app.
  - `app.py` — web app (risk input form → prediction)
  - `04_train_final_model.py` — trains and saves the final model
  - `01–10_*.py` — data prep, seed testing, threshold analysis, and validation steps
  - `description.md` / `USAGE_GUIDE.md` — technical write-up and usage guide
- **`main_tavi_training.csv`** — training data.

## Quick start

```bash
cd "try 12"
pip install flask pandas scikit-learn joblib numpy scipy matplotlib
python 04_train_final_model.py   # train + save the model
python app.py                    # start the web app
# open http://localhost:5000
```

## How the model was built

1. **Data prep** — cleaned missing values; log-transformed skewed measures where used.
2. **Feature selection** — tested combinations of candidate clinical features (calcium score, gradient, FCC, EuroSCORE II, ventricular mass); `FCC + Masa VS/SC` gave the best trade-off.
3. **Model choice** — logistic regression was chosen for interpretability and to avoid overfitting on a small event count (18 positives); more complex models didn't justify the added risk of overfit.
4. **Thresholding** — multiple presets let a clinician trade recall vs. accuracy explicitly, rather than baking in one operating point.

## Notes & limitations

This is a student/research project, not a certified clinical tool. It was trained on a small dataset from a single source; the numbers above describe performance on that data and should not be read as validated for real-world clinical decision-making. See `try 12/description.md` for the full methodology and caveats.

## Tech

`Python` · `scikit-learn` · `pandas` · `Flask` · `joblib`
