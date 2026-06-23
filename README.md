# Larval Population Prediction using Temporal Fusion Transformer (TFT)

A deep learning pipeline for predicting weekly larval infestation counts in agricultural fields using a Temporal Fusion Transformer built with PyTorch. The model combines multi-horizon probabilistic forecasting with interpretable attention mechanisms to support early pest intervention.

---

## Problem Statement

Larval infestations cause significant crop damage if not detected early. This project forecasts weekly larva counts across a 4-week horizon using historical weather data and agricultural covariates, enabling proactive field intervention.

---

## Dataset

- Weekly larva count records across multiple sowing types (early, normal, late)
- Meteorological features: temperature, rainfall, humidity, wind speed, and derived weather variables
- Date format inconsistencies between datasets were resolved by aligning on a standardised calendar index

---

## Data Preprocessing

- Merged larva count records with meteorological data on a common weekly time index
- Applied `log1p` transformation on the target variable (`larva_count`) to handle right-skewed count distributions
- Normalised all continuous features using per-feature `StandardScaler`
- Label-encoded `sowing_type` as a static categorical covariate
- Built sliding window samples (encoder: 24 weeks lookback, decoder: 4 weeks ahead) per sowing group

---

## Model Architecture

The TFT follows an encoder–decoder structure with the following components:

| Component | Details |
|---|---|
| Variable Selection Networks (VSN) | Learned soft feature weights per time step |
| LSTM Encoder–Decoder | 2-layer stacked, initialised from static covariate embeddings |
| Gated Residual Networks (GRN) | Skip connections for stable gradient flow |
| Static Enrichment | Sowing type context injected at every time step |
| Interpretable Multi-Head Attention | Causal mask, shared value projection across heads |
| Output | P10 / P50 / P90 quantile forecasts per decoder step |

---

## Training

- **Loss:** Combined quantile (pinball) loss + Huber loss on P50
- **Optimiser:** Adam (`lr=1e-3`)
- **Scheduler:** CosineAnnealingLR
- **Regularisation:** Dropout (0.1), gradient clipping (1.0), early stopping (patience=20)
- **Epochs:** 80

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE / RMSE | Point forecast accuracy on P50 |
| R² | Explained variance |
| Directional Accuracy | Correct up/down movement prediction |
| Tolerance Accuracy | % predictions within ±1 larva of actual |
| P10–P90 Coverage | % actuals falling within the prediction interval |

Benchmarked against ARIMA, SARIMAX, and Gradient Boosting baselines.

---

## Tech Stack

`Python` `PyTorch` `Pandas` `NumPy` `Scikit-learn` `Matplotlib`

---

## Project Structure
```
├── TFT_Predicton_model1__1_.ipynb   # Full pipeline: preprocessing → training → evaluation
└── README.md
```


