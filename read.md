# Soil Moisture Forecasting with ConvLSTM (t+1)

This project implements a **ConvLSTM-based deep learning framework** to forecast **daily soil moisture at 30 cm depth (sm_30cm)** using multivariate time-series data from multiple field probes.

The final experiment focuses on **short-term prediction (t+1)**, i.e., predicting soil moisture for the **next day**, which is more stable and physically meaningful than longer-horizon forecasts for the available data size.

---

## 1. Problem Description

- **Task**:  
  Predict next-day soil moisture (`sm_30cm`) using past observations.

- **Input**:  
  Multivariate daily time series (8 variables).

- **Output**:  
  Soil moisture at 30 cm depth for the next day (t+1).

- **Model**:  
  ConvLSTM2D (treating the feature dimension as a spatial axis).

- **Data structure**:  
  Multiple CSV files, each containing data from one region, further divided by `probe_name`.

---

## 2. Input Variables

Each daily record contains the following **8 input variables**:

| Variable | Description |
|--------|------------|
| sm_30cm | Soil moisture at 30 cm depth (volumetric water content) |
| irrig_mm | Daily irrigation amount (mm) |
| IRRAD | Solar radiation |
| TMIN | Minimum temperature |
| TMAX | Maximum temperature |
| VAP | Vapor pressure |
| WIND | Wind speed |
| RAIN | Precipitation |

> **Note**  
> `sm_30cm` is used both as an **input (autoregressive feature)** and as the **prediction target**.

---

## 3. Model Input / Output Definition

### Input (X)

For each sample, the model receives:

- The past **`window` days** of observations
- Shape:
X.shape = (N_samples, window, 1, 8, 1)

This follows the ConvLSTM2D input format: (batch, time, rows, cols, channels)

Where:
- `window` = lookback length (e.g. 14 or 30 days)
- `8` = number of input variables

---

### Output (Y)

- Single-step prediction (**t+1**)
- Shape:

Y.shape = (N_samples, 1)

- Represents soil moisture (`sm_30cm`) on the **next day**.

---

## 4. Data Splitting Strategy

Data are split **chronologically** (no shuffling):

Train / Validation / Test = 70% / 15% / 15%


### Context Borrowing (Scheme A)

To ensure sufficient historical context:

- **Training**
  - Uses only its own data.
- **Validation**
  - Input windows may borrow the last `window` days from the **training split**.
- **Test**
  - Input windows may borrow the last `window` days from the **validation split**.

> **Important**  
> Prediction targets (`y`) are always strictly inside their respective split.  
> This prevents any future information leakage and reflects real-world forecasting conditions.

---

## 5. Model Architecture

The ConvLSTM model consists of:

1. **ConvLSTM2D layer**
2. **Batch Normalization**
3. **Dropout (0.2)**
4. **Fully connected layers**
5. **Linear output layer** for t+1 prediction

Key hyperparameters:

- Filters: `16` (or `8` for smaller models)
- Dropout: `0.2`
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

---

## 6. Training Strategy

- **Early stopping** based on validation loss
- **Learning rate reduction** when validation loss plateaus
- Maximum epochs: `150` (training usually stops earlier)

Saved during training:

- `history.csv` – training & validation loss per epoch
- `loss_train_val.png` – loss curves

---

## 7. Evaluation Outputs

For each **region / probe / window**, the following outputs are generated:

### Quantitative Results
- `summary_metrics.json`
  - MAE (t+1)
  - RMSE (t+1)
  - Number of samples in each split

### Visualizations
- `loss_train_val.png`  
  Training vs validation loss
- `test_true_vs_pred_tplus1.png`  
  Actual vs predicted soil moisture on the test set

### Tables
- `test_compare_tplus1.csv`  
  Date-wise comparison of true and predicted values

---

## 8. Forecasting the Next Day

After training, the model is used to predict soil moisture for **the day following the last available observation**:

- Input: last `window` days
- Output file:
  - `last_date_forecast_next_1_day.csv`

---


## 9. Key Observations

- **t+1 predictions are stable and meaningful** for the available data length.
- Longer-horizon forecasts tend to regress toward the mean.
- ConvLSTM can easily overfit small datasets; small models and early stopping are essential.

---

## 10. Notes

- This project focuses on **data-driven forecasting**.
- No future meteorological forecasts are used.
- Sudden extreme events (e.g. heavy irrigation or sensor anomalies) are not expected to be predicted accurately.



