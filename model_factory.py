import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import logging
from datetime import datetime
import os
import json
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import onnx
import tf2onnx
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.xgboost
import mlflow.lightgbm
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Union, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)
logger = logging.getLogger(__name__)

class AdvancedModelFactory:
    """
    Professional-grade model factory with advanced features:
    - Multiple advanced architectures (ANN, XGBoost, LightGBM)
    - Hyperparameter optimization
    - Ensemble methods
    - Advanced preprocessing
    - Comprehensive evaluation
    - Model interpretation
    - Performance monitoring
    - Deployment capabilities
    """
    
    def __init__(self, model_dir='models', mlflow_tracking_uri=None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.preprocessor = None
        self.models = {}
        self.metrics = {}
        self.feature_importance = {}
        self.explainer = None
        self.mlflow_tracking_uri = mlflow_tracking_uri or "file:./mlruns"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
    def create_preprocessor(self, X):
        """Advanced preprocessing pipeline with feature engineering"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def create_advanced_ann(self, input_dim):
        """Advanced ANN architecture with attention mechanism"""
        inputs = Input(shape=(input_dim,))
        
        # Attention layer
        attention = Attention()([inputs, inputs])
        
        # First block
        x = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(attention)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Skip connection 1
        skip1 = Dense(128, activation='relu')(x)
        
        # Second block
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Skip connection 2
        skip2 = Dense(64, activation='relu')(x)
        
        # Third block
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Concatenate skip connections
        x = Concatenate()([x, skip1, skip2])
        
        # Final layers
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Advanced optimizer with learning rate scheduling
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )
        
        # Compile with focal loss for imbalanced data
        model.compile(
            optimizer=optimizer,
            loss=BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn')
            ]
        )
        
        return model
    
    def create_xgboost_model(self, params: Optional[Dict] = None) -> xgb.XGBClassifier:
        """Create advanced XGBoost model with optimized parameters"""
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'random_state': 42,
            'tree_method': 'hist',
            'enable_categorical': True
        }
        
        if params:
            default_params.update(params)
            
        return xgb.XGBClassifier(**default_params)
    
    def create_lightgbm_model(self, params: Optional[Dict] = None) -> lgb.LGBMClassifier:
        """Create advanced LightGBM model with optimized parameters"""
        default_params = {
            'objective': 'binary',
            'metric': 'aucpr',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True,
            'reg_alpha': 0.1,
            'reg_lambda': 1
        }
        
        if params:
            default_params.update(params)
            
        return lgb.LGBMClassifier(**default_params)
    
    def get_callbacks(self, model_name):
        """Advanced training callbacks with monitoring"""
        return [
            EarlyStopping(
                monitor='val_pr_auc',
                patience=20,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_pr_auc',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'{self.model_dir}/{model_name}_best.h5',
                monitor='val_pr_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            TensorBoard(
                log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
    
    def train_ensemble(self, X, y, n_splits=5):
        """Train ensemble of models with advanced techniques"""
        # Preprocess data
        self.preprocessor = self.create_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store metrics
        metrics = {
            'auc': [], 'pr_auc': [], 'accuracy': [], 'precision': [], 
            'recall': [], 'f1': [], 'confusion_matrix': [], 'classification_report': []
        }
        
        # Start MLflow experiment
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("model_types", ["ANN", "XGBoost", "LightGBM"])
            
            # Train models
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled), 1):
                logger.info(f"Training fold {fold}/{n_splits}")
                
                X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
                y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
                
                # Train ANN
                ann_model = self.create_advanced_ann(X_train.shape[1])
                ann_history = ann_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=32,
                    callbacks=self.get_callbacks(f'fold_{fold}_ann'),
                    verbose=1
                )
                
                # Train XGBoost if available
                try:
                    xgb_model = self.create_xgboost_model()
                    xgb_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                    self.models[f'fold_{fold}_xgb'] = xgb_model
                except ImportError:
                    logger.warning("XGBoost not available, skipping...")
                
                # Train LightGBM if available
                try:
                    lgb_model = self.create_lightgbm_model()
                    lgb_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                    self.models[f'fold_{fold}_lgb'] = lgb_model
                except ImportError:
                    logger.warning("LightGBM not available, skipping...")
                
                # Evaluate models
                for model_name, model in self.models.items():
                    if fold in model_name:
                        y_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
                        y_pred_binary = (y_pred > 0.5).astype(int)
                        
                        # Calculate metrics
                        metrics['auc'].append(roc_auc_score(y_val, y_pred))
                        metrics['pr_auc'].append(average_precision_score(y_val, y_pred))
                        metrics['accuracy'].append(accuracy_score(y_val, y_pred_binary))
                        metrics['precision'].append(precision_score(y_val, y_pred_binary))
                        metrics['recall'].append(recall_score(y_val, y_pred_binary))
                        metrics['f1'].append(f1_score(y_val, y_pred_binary))
                        metrics['confusion_matrix'].append(confusion_matrix(y_val, y_pred_binary))
                        metrics['classification_report'].append(classification_report(y_val, y_pred_binary))
                        
                        # Log metrics to MLflow
                        mlflow.log_metrics({
                            f"{model_name}_auc": metrics['auc'][-1],
                            f"{model_name}_pr_auc": metrics['pr_auc'][-1],
                            f"{model_name}_accuracy": metrics['accuracy'][-1],
                            f"{model_name}_precision": metrics['precision'][-1],
                            f"{model_name}_recall": metrics['recall'][-1],
                            f"{model_name}_f1": metrics['f1'][-1]
                        })
                        
                        # Calculate feature importance
                        if hasattr(model, 'feature_importances_'):
                            self.feature_importance[model_name] = model.feature_importances_
                        else:
                            explainer = shap.KernelExplainer(model.predict, X_train[:100])
                            shap_values = explainer.shap_values(X_val[:100])
                            self.feature_importance[model_name] = np.abs(shap_values).mean(0)
                
                # Save models
                self.models[f'fold_{fold}_ann'] = ann_model
        
        # Calculate final metrics
        self.metrics = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            for metric, values in metrics.items()
            if metric not in ['confusion_matrix', 'classification_report']
        }
        
        # Add confusion matrix and classification report
        self.metrics['confusion_matrix'] = np.mean(metrics['confusion_matrix'], axis=0)
        self.metrics['classification_report'] = metrics['classification_report'][-1]
        
        # Save ensemble and metrics
        self.save_ensemble()
        
        # Generate performance plots
        self.generate_performance_plots()
        
        # Log artifacts to MLflow
        mlflow.log_artifacts(self.model_dir)
        
        return self.metrics
    
    def generate_performance_plots(self):
        """Generate comprehensive performance plots using Plotly"""
        try:
            plots_dir = os.path.join(self.model_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('ROC Curves', 'Feature Importance', 'Confusion Matrix', 'Precision-Recall Curve')
            )
            
            # ROC Curves
            for model_name in self.models:
                if 'ann' in model_name:
                    fpr, tpr, _ = roc_curve(self.metrics['y_true'], self.metrics['y_pred'])
                    fig.add_trace(
                        go.Scatter(x=fpr, y=tpr, name=f'{model_name} ROC'),
                        row=1, col=1
                    )
            
            # Feature Importance
            for model_name, importance in self.feature_importance.items():
                fig.add_trace(
                    go.Bar(x=list(range(len(importance))), y=importance, name=f'{model_name} Importance'),
                    row=1, col=2
                )
            
            # Confusion Matrix
            fig.add_trace(
                go.Heatmap(
                    z=self.metrics['confusion_matrix'],
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    colorscale='Blues'
                ),
                row=2, col=1
            )
            
            # Precision-Recall Curve
            for model_name in self.models:
                if 'ann' in model_name:
                    precision, recall, _ = precision_recall_curve(self.metrics['y_true'], self.metrics['y_pred'])
                    fig.add_trace(
                        go.Scatter(x=recall, y=precision, name=f'{model_name} PR'),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                height=1000,
                width=1200,
                title_text="Model Performance Analysis",
                showlegend=True
            )
            
            # Save plot
            fig.write_html(os.path.join(plots_dir, 'performance_analysis.html'))
            
            # Generate additional plots
            self._generate_model_comparison_plot()
            self._generate_feature_importance_plot()
            self._generate_calibration_plot()
            
        except Exception as e:
            logger.error(f"Error generating performance plots: {str(e)}")
    
    def _generate_model_comparison_plot(self):
        """Generate model comparison plot"""
        metrics = ['auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1']
        models = list(self.models.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [self.metrics[metric]['mean'] for _ in models]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Metric Value",
            barmode='group'
        )
        
        fig.write_html(os.path.join(self.model_dir, 'plots', 'model_comparison.html'))
    
    def _generate_feature_importance_plot(self):
        """Generate feature importance plot"""
        fig = go.Figure()
        
        for model_name, importance in self.feature_importance.items():
            fig.add_trace(go.Bar(
                name=model_name,
                x=list(range(len(importance))),
                y=importance
            ))
        
        fig.update_layout(
            title="Feature Importance Comparison",
            xaxis_title="Features",
            yaxis_title="Importance",
            barmode='group'
        )
        
        fig.write_html(os.path.join(self.model_dir, 'plots', 'feature_importance.html'))
    
    def _generate_calibration_plot(self):
        """Generate calibration plot"""
        fig = go.Figure()
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                prob_true, prob_pred = calibration_curve(
                    self.metrics['y_true'],
                    model.predict_proba(self.metrics['X_test'])[:, 1],
                    n_bins=10
                )
                fig.add_trace(go.Scatter(
                    x=prob_pred,
                    y=prob_true,
                    name=model_name,
                    mode='lines+markers'
                ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Perfect Calibration',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title="Calibration Curves",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives"
        )
        
        fig.write_html(os.path.join(self.model_dir, 'plots', 'calibration.html'))
    
    def save_ensemble(self):
        """Save ensemble models and metrics"""
        try:
            # Save models
            for name, model in self.models.items():
                model.save(f'{self.model_dir}/{name}.h5')
            
            # Save preprocessor
            joblib.dump(self.preprocessor, f'{self.model_dir}/preprocessor.pkl')
            
            # Save metrics
            with open(f'{self.model_dir}/metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            # Save feature importance
            with open(f'{self.model_dir}/feature_importance.json', 'w') as f:
                json.dump({k: v.tolist() for k, v in self.feature_importance.items()}, f, indent=4)
            
            logger.info("Ensemble saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {str(e)}")
    
    @classmethod
    def load_ensemble(cls, model_dir):
        """Load trained ensemble with all components"""
        try:
            instance = cls(model_dir)
            
            # Load preprocessor
            instance.preprocessor = joblib.load(f'{model_dir}/preprocessor.pkl')
            
            # Load models
            for model_file in os.listdir(model_dir):
                if model_file.endswith('.h5') and 'fold' in model_file:
                    name = model_file.replace('.h5', '')
                    instance.models[name] = load_model(f'{model_dir}/{model_file}')
            
            # Load metrics
            with open(f'{model_dir}/metrics.json', 'r') as f:
                instance.metrics = json.load(f)
            
            # Load feature importance
            with open(f'{model_dir}/feature_importance.json', 'r') as f:
                instance.feature_importance = {
                    k: np.array(v) for k, v in json.load(f).items()
                }
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {str(e)}")
            return None
    
    def deploy_model(self, model_name: str, format: str = 'onnx') -> str:
        """Deploy model in specified format"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            deploy_dir = os.path.join(self.model_dir, 'deploy')
            os.makedirs(deploy_dir, exist_ok=True)
            
            if format == 'onnx':
                if isinstance(model, tf.keras.Model):
                    # Convert TensorFlow model to ONNX
                    onnx_model, _ = tf2onnx.convert.from_keras(model)
                    onnx.save(onnx_model, os.path.join(deploy_dir, f'{model_name}.onnx'))
                else:
                    # Convert scikit-learn model to ONNX
                    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
                    onnx_model = convert_sklearn(model, initial_types=initial_type)
                    onnx.save(onnx_model, os.path.join(deploy_dir, f'{model_name}.onnx'))
            
            elif format == 'tensorflow':
                if isinstance(model, tf.keras.Model):
                    model.save(os.path.join(deploy_dir, f'{model_name}.h5'))
                else:
                    raise ValueError("Only TensorFlow models can be saved in TensorFlow format")
            
            elif format == 'pickle':
                with open(os.path.join(deploy_dir, f'{model_name}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Save preprocessor
            joblib.dump(self.preprocessor, os.path.join(deploy_dir, 'preprocessor.pkl'))
            
            # Create deployment metadata
            metadata = {
                'model_name': model_name,
                'format': format,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics
            }
            
            with open(os.path.join(deploy_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model {model_name} deployed successfully in {format} format")
            return os.path.join(deploy_dir, f'{model_name}.{format}')
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise
    
    def create_api_endpoint(self, model_name: str, port: int = 5000) -> None:
        """Create Flask API endpoint for model serving"""
        try:
            from flask import Flask, request, jsonify
            
            app = Flask(__name__)
            
            @app.route('/predict', methods=['POST'])
            def predict():
                try:
                    data = request.get_json()
                    X = pd.DataFrame(data)
                    X_processed = self.preprocessor.transform(X)
                    model = self.models[model_name]
                    
                    if hasattr(model, 'predict_proba'):
                        predictions = model.predict_proba(X_processed)[:, 1]
                    else:
                        predictions = model.predict(X_processed)
                    
                    return jsonify({
                        'predictions': predictions.tolist(),
                        'model': model_name,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    return jsonify({'error': str(e)}), 400
            
            app.run(host='0.0.0.0', port=port)
            
        except ImportError:
            logger.error("Flask not installed. Please install Flask to create API endpoint.")
            raise

def main():
    """Main training function with comprehensive error handling"""
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv('data/autism_screening_dataset.csv')
        
        # Prepare data
        X = df.drop(columns=['Autism_Diagnosis'])
        y = df['Autism_Diagnosis'].map({'YES': 1, 'NO': 0})
        
        # Initialize and train
        logger.info("Initializing Advanced Model Factory...")
        factory = AdvancedModelFactory()
        
        logger.info("Starting training...")
        metrics = factory.train_ensemble(X, y)
        
        # Log results
        logger.info("\nTraining completed. Final metrics:")
        for metric, values in metrics.items():
            if isinstance(values, dict):
                logger.info(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
            elif metric == 'classification_report':
                logger.info(f"\n{metric}:\n{values}")
        
        # Deploy best model
        best_model = max(factory.models.items(), key=lambda x: metrics['auc']['mean'])
        factory.deploy_model(best_model[0], format='onnx')
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 