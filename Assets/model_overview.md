# Model Comparison Results

## Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| LightGBM | 0.96 | 0.40 | 0.00 | 0.00 | **0.64** |
| Gradient Boosting | 0.96 | 0.35 | 0.00 | 0.00 | **0.64** |
| AdaBoost | 0.96 | 0.00 | 0.00 | 0.00 | **0.64** |
| Random Forest | 0.96 | 0.00 | 0.00 | 0.00 | 0.63 |
| MLPClassifier | 0.96 | 0.00 | 0.00 | 0.00 | 0.63 |
| XGBoost | 0.96 | 0.80 | 0.00 | 0.00 | 0.63 |
| Logistic Regression | 0.96 | 0.00 | 0.00 | 0.00 | 0.62 |
| Naive Bayes | 0.90 | 0.07 | 0.13 | 0.09 | 0.61 |
| K-Nearest Neighbors | 0.96 | 0.07 | 0.00 | 0.00 | 0.52 |
| Dummy Classifier | 0.96 | 0.00 | 0.00 | 0.00 | 0.50 |

## Top 3 Models by ROC-AUC

1. **LightGBM**: 0.6387
2. **Gradient Boosting**: 0.6387
3. **AdaBoost**: 0.6356

---

### Key Observations

- All models achieved high accuracy (0.90-0.96), suggesting class imbalance in the dataset
- LightGBM and Gradient Boosting tied for the best ROC-AUC score (0.6387)
- Most models show poor recall and F1-scores, indicating difficulty in identifying positive class
- Dummy Classifier baseline: 0.50 ROC-AUC
