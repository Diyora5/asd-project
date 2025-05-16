import pandas as pd
from scipy.io import arff
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import logging
import time

# Optional imports for advanced models
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_fix_data(filepath):
    """Load data and fix inconsistencies before parsing"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the problematic value before parsing
    content = content.replace('HealthCare Professional', 'Health Care Professional')
    
    # Write to temporary file
    temp_path = 'temp_fixed.arff'
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        data = arff.loadarff(temp_path)
        df = pd.DataFrame(data[0])
        
        # Decode byte strings
        for col in df.select_dtypes([object]):
            df[col] = df[col].str.decode('utf-8')
        
        return df
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def preprocess_data(df):
    """Preprocess the dataset using 'Autism_Diagnosis' as target"""
    target_col = 'Autism_Diagnosis'
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Encode target (should already be 0/1 but ensure it's integer)
    df[target_col] = df[target_col].astype(int)
    
    # Encode categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    return df, target_col

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with multiple metrics"""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    avg_precision = average_precision_score(y_test, y_proba) if y_proba is not None else None

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'average_precision': avg_precision
    }

def main():
    os.makedirs('models', exist_ok=True)

    try:
        # Load and preprocess data
        df = load_and_fix_data('data/autism_screening_dataset.arff')
        df, target_col = preprocess_data(df)

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Oversampling to balance classes
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define models and hyperparameter grids
        models = {}
        param_grids = {}

        # Decision Tree
        models['DecisionTree'] = DecisionTreeClassifier(random_state=42)
        param_grids['DecisionTree'] = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Random Forest
        models['RandomForest'] = RandomForestClassifier(random_state=42)
        param_grids['RandomForest'] = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'class_weight': [None, 'balanced']
        }

        # Logistic Regression
        models['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=1000)
        param_grids['LogisticRegression'] = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

        # XGBoost (if available)
        if xgb is not None:
            models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            param_grids['XGBoost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0]
            }

        # LightGBM (if available)
        if lgb is not None:
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42)
            param_grids['LightGBM'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }

        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            logging.info(f"Starting training for {name}...")
            param_grid = param_grids.get(name, {})
            if param_grid:
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=10, scoring='roc_auc', cv=skf, random_state=42, n_jobs=-1, verbose=1)
                start_time = time.time()
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                elapsed = time.time() - start_time
                logging.info(f"Best params for {name}: {search.best_params_}")
                logging.info(f"Training time for {name}: {elapsed:.2f} seconds")
            else:
                start_time = time.time()
                model.fit(X_train, y_train)
                best_model = model
                elapsed = time.time() - start_time
                logging.info(f"Training time for {name}: {elapsed:.2f} seconds")

            # Evaluate
            metrics = evaluate_model(best_model, X_test, y_test)
            logging.info(f"{name} evaluation metrics:")
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    logging.info(f"  {metric_name}: {metric_value:.4f}")

            # Save model
            model_filename = f"models/{name.lower()}_best.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(best_model, f)
            logging.info(f"Saved best {name} model to {model_filename}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
