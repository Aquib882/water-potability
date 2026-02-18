# 📊 Power BI Dashboard — Step-by-Step Guide
## Water Potability Prediction Project

---

## Data Sources (CSV Files from models/ folder)

| File | Use |
|------|-----|
| `balanced_dataset.csv` | Raw data explorer, distributions, scatter plots |
| `test_predictions.csv` | Model predictions, confusion analysis |
| `final_metrics.csv` | Model comparison bar charts |
| `feature_importance.csv` | Feature ranking visuals |

---

## Step 1: Import Data into Power BI

1. Open **Power BI Desktop**
2. Click **Home → Get Data → Text/CSV**
3. Import all 4 CSV files listed above
4. For each file: confirm delimiter is comma, click **Load**
5. Go to **Model view** — no relationships needed (flat tables)

---

## Step 2: Page 1 — Dataset Overview

**Title:** "Water Quality Dataset Explorer"

### Cards (top row)
- **Card 1:** Total Samples → `COUNTROWS(balanced_dataset)`
- **Card 2:** Potable % → `DIVIDE(COUNTROWS(FILTER(balanced_dataset, balanced_dataset[Potability]=1)), COUNTROWS(balanced_dataset), 0) * 100`
- **Card 3:** Not Potable % → complement of above
- **Card 4:** Total Features → Static value: 9

### Donut Chart
- Values: `Potability` column from balanced_dataset
- Rename values: 0 = "Not Potable", 1 = "Potable"
- Colors: Blue (#2196F3) for Not Potable, Green (#4CAF50) for Potable

### Histogram (repeat for each feature)
- Use **Column Chart** visual
- X-axis: Binned feature value (create bins in Power Query)
- Y-axis: Count of rows
- Add **Slicer** on Potability column to filter by class

### Correlation Table
- Use a **Matrix** visual
- Import the correlation values manually or compute in Python visual

---

## Step 3: Page 2 — Model Performance

**Title:** "ML Model Comparison"

### Clustered Bar Chart
- Data: `final_metrics.csv`
- X-axis: Model name
- Y-axis: Metric value
- Legend: Metric type (Accuracy, Precision, Recall, F1-Score)
- Colors: Each metric a different color

### Gauge Chart (Best Model)
- Value: F1-Score of Gradient Boosting = 0.893
- Min: 0, Max: 1, Target: 0.9
- Label: "Best Model F1-Score"

### Table Visual
- All columns from final_metrics.csv
- Conditional formatting on F1-Score column (green gradient)
- Sort descending by F1-Score

### KPI Cards (one per metric for best model)
- Accuracy: 89.25%
- Precision: 89.05%
- Recall: 89.50%
- F1-Score: 89.28%

---

## Step 4: Page 3 — Predictions Explorer

**Title:** "Model Predictions Deep-Dive"

### Matrix Visual (Confusion Matrix)
- Rows: `Actual` column from test_predictions.csv
- Columns: `Predicted` column
- Values: Count of rows
- Format: Conditional color formatting
  - High values on diagonal = Green
  - Off-diagonal = Red

### Histogram — Prediction Probabilities
- X-axis: `Probability_Potable` (bin by 0.1 intervals)
- Y-axis: Count
- Color by: `Potability` (actual class)

### Table — Sample Predictions
- Columns: All features + Actual + Predicted + Probability_Potable
- Filter: Add slicers for Actual and Predicted
- Conditional formatting on Probability column

### Scatter Plot — Probability vs pH
- X-axis: pH
- Y-axis: Probability_Potable
- Color: Predicted (0 or 1)

---

## Step 5: Page 4 — Feature Importance

**Title:** "What Makes Water Potable?"

### Horizontal Bar Chart (Main Visual)
- Data: `feature_importance.csv`
- Y-axis: Feature name (sorted by importance)
- X-axis: Importance score
- Color: Gradient from low (light blue) to high (dark blue)
- Data labels: Enabled, showing importance value

### Waterfall Chart (Alternative)
- Same data as above
- Shows additive contribution of each feature

### Scatter Plots (Feature vs Potability)
- Use balanced_dataset.csv
- Plot Solids vs Potability, pH vs Potability, Sulfate vs Potability
- Color by Potability class

### WHO Standards Comparison Table
| Feature | Model Importance | WHO Standard |
|---------|-----------------|--------------|
| Solids | Highest | < 500 ppm |
| Conductivity | 2nd | < 400 μS/cm |
| Sulfate | 3rd | < 250 mg/L |
| pH | 4th | 6.5 – 8.5 |

---

## Step 6: Styling & Formatting

### Theme
- Background: White (#FFFFFF) or Light Blue (#E3F2FD)
- Primary accent: #1565C0 (Dark Blue)
- Success color: #4CAF50 (Green)
- Alert color: #F44336 (Red)

### Typography
- Title font: Segoe UI, 20pt, Bold
- Labels: Segoe UI, 11pt
- Values: Segoe UI, 14pt, Bold

### Navigation
- Add page navigation buttons in the sidebar
- Use icons: 📊 🔍 💧 📈

### Slicers (add to all pages)
- Potability class slicer (dropdown or toggle)
- pH range slicer (numeric range)

---

## Step 7: Publish (Optional)

1. **File → Publish → Power BI Service**
2. Sign in with Microsoft account (free)
3. Select workspace → Publish
4. Share the dashboard link with stakeholders

---

## DAX Measures Reference

```dax
-- Accuracy
Accuracy = 
DIVIDE(
    COUNTROWS(FILTER(test_predictions, test_predictions[Actual] = test_predictions[Predicted])),
    COUNTROWS(test_predictions)
) * 100

-- Potable count
Potable_Count = COUNTROWS(FILTER(balanced_dataset, balanced_dataset[Potability] = 1))

-- Not Potable count  
Not_Potable_Count = COUNTROWS(FILTER(balanced_dataset, balanced_dataset[Potability] = 0))

-- Average Probability
Avg_Probability = AVERAGE(test_predictions[Probability_Potable])

-- True Positives
TP = COUNTROWS(FILTER(test_predictions, test_predictions[Actual] = 1 && test_predictions[Predicted] = 1))

-- False Positives
FP = COUNTROWS(FILTER(test_predictions, test_predictions[Actual] = 0 && test_predictions[Predicted] = 1))

-- Precision (manual)
Precision_Manual = DIVIDE([TP], [TP] + [FP])
```
