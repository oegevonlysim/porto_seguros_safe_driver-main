"""
LightGBM Classifier Script for Porto Seguro Safe Driver Prediction

This script implements a quick LightGBM classifier using default parameters
on the Porto Seguro Safe Driver Prediction dataset. It includes functionality
to train the model, evaluate performance, and visualize top feature importances.

Author: GitHub Copilot
Date: 2025-11-12
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from lightgbm import LGBMClassifier
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_data(data_path='data/train.csv', use_mock=False):
    """
    Load the Porto Seguro dataset from the specified path.
    
    Parameters:
    -----------
    data_path : str
        Path to the training data CSV file
    use_mock : bool
        If True, creates a mock dataset for demonstration purposes
        
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    feature_names : list
        List of feature column names
    """
    print("=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)
    
    # Check if the actual data file exists
    if not use_mock and os.path.exists(data_path):
        print(f"✓ Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✓ Data loaded successfully!")
        print(f"  - Dataset shape: {df.shape}")
        print(f"  - Columns: {df.shape[1]}")
        print(f"  - Rows: {df.shape[0]}")
        
        # Separate features and target
        # Drop 'id' column as it's not a feature
        X = df.drop(['id', 'target'], axis=1, errors='ignore')
        y = df['target']
        
        print(f"\n✓ Using ACTUAL Porto Seguro dataset")
        
    else:
        # Create a mock dataset for demonstration
        print("⚠ Actual dataset not found. Creating MOCK dataset for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate random features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate target variable (binary classification)
        y = pd.Series(np.random.randint(0, 2, n_samples), name='target')
        
        print(f"✓ Mock dataset created!")
        print(f"  - Dataset shape: {X.shape}")
        print(f"  - Features: {n_features}")
        print(f"  - Samples: {n_samples}")
        print(f"\n⚠ NOTE: This is a MOCK dataset, not the actual Porto Seguro data!")
    
    feature_names = X.columns.tolist()
    
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print(f"Class balance: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y, feature_names


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Training and testing splits
    """
    print("\n" + "=" * 80)
    print("STEP 2: Preparing Data")
    print("=" * 80)
    
    # Handle missing values by filling with -999 (common strategy for tree-based models)
    print("✓ Handling missing values...")
    X_filled = X.fillna(-999)
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"  - Filled {missing_count} missing values with -999")
    else:
        print(f"  - No missing values found")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_filled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✓ Data split completed!")
    print(f"  - Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"  - Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    print(f"  - Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a LightGBM classifier with default parameters.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target variable
        
    Returns:
    --------
    model : LGBMClassifier
        Trained LightGBM model
    """
    print("\n" + "=" * 80)
    print("STEP 3: Training LightGBM Classifier")
    print("=" * 80)
    
    print("✓ Initializing LGBMClassifier with default parameters...")
    print("  - Using default LightGBM hyperparameters")
    print("  - This is a quick baseline model")
    
    # Initialize the model with default parameters
    # Setting verbose=-1 to suppress training output
    model = LGBMClassifier(random_state=42, verbose=-1)
    
    print("\n✓ Training the model...")
    model.fit(X_train, y_train)
    
    print("✓ Model training completed!")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the trained model on both training and test sets.
    
    Parameters:
    -----------
    model : LGBMClassifier
        Trained model
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target variable
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test target variable
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 80)
    print("STEP 4: Evaluating Model Performance")
    print("=" * 80)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Predict probabilities for ROC-AUC calculation
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    print("\n📊 Performance Metrics:")
    print("-" * 80)
    print(f"Training Set:")
    print(f"  - Accuracy:  {train_accuracy:.4f}")
    print(f"  - ROC-AUC:   {train_roc_auc:.4f}")
    print(f"\nTest Set:")
    print(f"  - Accuracy:  {test_accuracy:.4f}")
    print(f"  - ROC-AUC:   {test_roc_auc:.4f}")
    
    print("\n📋 Classification Report (Test Set):")
    print("-" * 80)
    print(classification_report(y_test, y_test_pred))
    
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc
    }
    
    return metrics


def plot_feature_importance(model, feature_names, top_n=20, save_path='feature_importance.png'):
    """
    Plot the top N feature importances from the trained model.
    
    Parameters:
    -----------
    model : LGBMClassifier
        Trained LightGBM model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display (default: 20)
    save_path : str
        Path to save the plot image (default: 'feature_importance.png')
    """
    print("\n" + "=" * 80)
    print("STEP 5: Plotting Feature Importances")
    print("=" * 80)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for easier manipulation
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = feature_importance_df.head(top_n)
    
    print(f"\n✓ Top {top_n} Most Important Features:")
    print("-" * 80)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:30s} : {row['importance']:.2f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    
    # Customize the plot
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importances - LightGBM Classifier', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()  # Highest importance at the top
    
    # Add value labels on the bars
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        plt.text(value, i, f' {value:.2f}', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Feature importance plot saved to: {save_path}")
    
    # Display the plot
    plt.show()
    print("✓ Plot displayed!")


def main():
    """
    Main function to orchestrate the entire workflow:
    1. Load data
    2. Prepare data (split and handle missing values)
    3. Train LightGBM classifier
    4. Evaluate model performance
    5. Plot feature importances
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "LightGBM Classifier - Porto Seguro Dataset" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Step 1: Load data
    # First try to load actual data, fall back to mock if not available
    data_path = 'data/train.csv'
    use_mock = not os.path.exists(data_path)
    
    X, y, feature_names = load_data(data_path=data_path, use_mock=use_mock)
    
    # Step 2: Prepare data (train-test split)
    X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Train the model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate the model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Step 5: Plot feature importances
    plot_feature_importance(model, feature_names, top_n=20, save_path='feature_importance.png')
    
    print("\n" + "=" * 80)
    print("✓ All steps completed successfully!")
    print("=" * 80)
    print("\n📝 Summary:")
    print(f"  - Model: LightGBM Classifier (default parameters)")
    print(f"  - Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  - Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"  - Feature importance plot saved: feature_importance.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
