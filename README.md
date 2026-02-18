# 💧 Water Potability Prediction — ML Project

> Predict whether water is safe for consumption using machine learning.

## 🏆 Results

| Model                     | Accuracy | Precision | Recall | F1-Score |
|---------------------------|----------|-----------|--------|----------|
| SVC (Tuned)               | 73.25%   | 72.04%    | 76.00% | 73.97%   |
| Random Forest (Tuned)     | 89.25%   | 90.46%    | 87.75% | 89.09%   |
| KNN                       | 63.75%   | 62.79%    | 67.50% | 65.06%   |
| Decision Tree             | 83.50%   | 80.04%    | 89.25% | 84.40%   |
| Gradient Boosting ⭐       | 89.25%   | 89.05%    | 89.50% | 89.28%   |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train_model.py        # Train models
python app.py                # Start web app → http://localhost:5000
jupyter notebook Water_Potability_Prediction.ipynb
```

## 🌐 Deployment

### Render.com (Free)
1. Push to GitHub
2. New Web Service → connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app:app`

### Railway.app
Same steps — set start command to `gunicorn app:app`

## 📊 Power BI
Import CSVs from models/ folder. See PowerBI_Dashboard_Guide.md for full setup.

## API Example
```python
import requests
r = requests.post("http://localhost:5000/predict", json={
    "ph": 7.0, "Hardness": 196.0, "Solids": 22000.0,
    "Chloramines": 7.1, "Sulfate": 333.0, "Conductivity": 426.0,
    "Organic_carbon": 14.0, "Trihalomethanes": 66.0, "Turbidity": 3.97
})
print(r.json())  # {'prediction': 0, 'potable': False, 'probability': 1.22, ...}
```
