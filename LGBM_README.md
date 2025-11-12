# LightGBM Classifier for Porto Seguro Safe Driver Prediction

## Overview

This script (`lgbm_classifier.py`) implements a quick baseline LightGBM classifier for the Porto Seguro Safe Driver Prediction dataset. It uses default LightGBM parameters to train a binary classification model and provides feature importance visualization.

## Features

- ✅ Automatic data loading from Porto Seguro dataset
- ✅ Fallback to mock dataset if actual data is unavailable
- ✅ Data preprocessing (missing value handling, train-test split)
- ✅ LightGBM classifier training with default parameters
- ✅ Comprehensive model evaluation (Accuracy, ROC-AUC)
- ✅ Beautiful feature importance visualization
- ✅ Thoroughly documented code

## Requirements

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `lightgbm==4.5.0` - Gradient boosting framework
- `pandas==2.2.2` - Data manipulation
- `scikit-learn==1.5.1` - Machine learning utilities
- `matplotlib==3.9.2` - Plotting
- `seaborn==0.13.2` - Statistical visualization

## Usage

### Basic Usage

Simply run the script from the repository root:

```bash
python lgbm_classifier.py
```

### What the Script Does

The script follows these steps:

1. **Load Data**: Reads the Porto Seguro training data from `data/train.csv`
   - If the data file is not found, it creates a mock dataset for demonstration
   
2. **Prepare Data**: 
   - Handles missing values (fills with -999, a common strategy for tree-based models)
   - Splits into 80% training and 20% test sets
   - Maintains class balance with stratified sampling
   
3. **Train Model**:
   - Initializes LGBMClassifier with default parameters
   - Trains on the training set
   
4. **Evaluate Performance**:
   - Calculates accuracy and ROC-AUC for both train and test sets
   - Provides detailed classification report
   
5. **Visualize Results**:
   - Generates a feature importance plot showing the top 20 most important features
   - Saves the plot as `feature_importance.png`

## Expected Output

When you run the script, you'll see detailed output for each step:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               LightGBM Classifier - Porto Seguro Dataset                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

STEP 1: Loading Data
✓ Loading data from: data/train.csv
✓ Data loaded successfully!
  - Dataset shape: (595212, 59)
  ...

STEP 2: Preparing Data
✓ Handling missing values...
✓ Data split completed!
  ...

STEP 3: Training LightGBM Classifier
✓ Initializing LGBMClassifier with default parameters...
✓ Training the model...
✓ Model training completed!

STEP 4: Evaluating Model Performance
📊 Performance Metrics:
  - Test Accuracy: 0.9635
  - Test ROC-AUC: 0.6360
  ...

STEP 5: Plotting Feature Importances
✓ Top 20 Most Important Features:
  ps_car_13      : 234.00
  ps_reg_03      : 195.00
  ...
✓ Feature importance plot saved to: feature_importance.png
```

## Output Files

The script generates:
- `feature_importance.png` - Horizontal bar chart showing the top 20 most important features

## Data Requirements

### Using Actual Porto Seguro Data

The script expects the Porto Seguro training data to be located at:
```
data/train.csv
```

This file should contain the Porto Seguro Safe Driver Prediction dataset with 59 columns including:
- `id` - Unique identifier
- `target` - Binary target variable (0 = no claim, 1 = claim filed)
- 57 feature columns (ps_ind_*, ps_reg_*, ps_car_*, ps_calc_*)

### Using Mock Data

If `data/train.csv` is not found, the script automatically generates a mock dataset:
- 1,000 samples
- 20 random features
- Binary target variable
- This allows you to test the script functionality without the actual data

## Code Structure

The script is organized into well-documented functions:

- `load_data()` - Load Porto Seguro data or generate mock data
- `prepare_data()` - Preprocess and split data
- `train_model()` - Train LightGBM classifier
- `evaluate_model()` - Evaluate model performance
- `plot_feature_importance()` - Visualize feature importances
- `main()` - Orchestrate the entire workflow

## Performance Notes

### Model Performance
With default parameters on the actual Porto Seguro dataset:
- Test Accuracy: ~96.35%
- Test ROC-AUC: ~0.636

Note: The high accuracy is partly due to severe class imbalance (96.4% of samples are class 0).

### Top Important Features
The script identifies the most predictive features, typically including:
- `ps_car_13` - Continuous car-related feature
- `ps_reg_03` - Regional numeric feature
- `ps_ind_03` - Individual ordinal feature
- `ps_ind_15` - Individual continuous feature
- `ps_car_14` - Continuous car-related feature

## Customization

You can easily customize the script by modifying parameters in the `main()` function:

```python
# Change train-test split ratio
X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=0.3)  # 70-30 split

# Change number of top features to display
plot_feature_importance(model, feature_names, top_n=30)  # Show top 30

# Change output file name
plot_feature_importance(model, feature_names, save_path='my_plot.png')
```

## Model Improvements

This is a baseline model using default parameters. For better performance, consider:

1. **Hyperparameter tuning**: Use GridSearchCV or Optuna to optimize parameters
2. **Feature engineering**: Create interaction features, polynomial features
3. **Handle class imbalance**: Use class weights, SMOTE, or undersampling
4. **Feature selection**: Remove low-importance or calc_* features
5. **Ensemble methods**: Combine multiple models

## Troubleshooting

### "ModuleNotFoundError: No module named 'lightgbm'"
Install dependencies: `pip install -r requirements.txt`

### "FileNotFoundError: data/train.csv not found"
The script will automatically use mock data. To use actual data, extract the Porto Seguro dataset to `data/train.csv`

### Plot doesn't display
If running in a headless environment, the plot will still be saved to `feature_importance.png` even if it can't be displayed.

## License

This script is part of the Porto Seguro Safe Driver Prediction project. See the repository LICENSE for details.

## References

- [Porto Seguro's Safe Driver Prediction on Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
