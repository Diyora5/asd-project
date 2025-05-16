import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging
import json
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ann_training.log')
    ]
)
logger = logging.getLogger(__name__)

class AdvancedANNModel:
    """
    Advanced Artificial Neural Network for ASD prediction with professional features:
    - Batch Normalization
    - Dropout
    - Learning Rate Scheduling
    - Early Stopping
    - Cross-Validation
    - Comprehensive Evaluation
    """
    
    def __init__(self, input_dim, model_path='models/advanced_ann'):
        self.input_dim = input_dim
        self.model_path = model_path
        self.preprocessor = None
        self.history = None
        self.model = None
        self.metrics = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def create_preprocessor(self, X):
        """Create preprocessing pipeline for both numeric and categorical features"""
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def create_model(self, input_dim):
        """Create advanced ANN architecture with professional features"""
        model = Sequential([
            # Input layer with batch normalization
            Dense(256, activation='relu', input_dim=input_dim,
                  kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers with increasing complexity
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with advanced optimizer settings
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def get_callbacks(self):
        """Configure advanced training callbacks"""
        return [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_auc',
                patience=20,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction on plateau
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                mode='max',
                verbose=1
            ),
            
            # Model checkpointing
            ModelCheckpoint(
                filepath=f'{self.model_path}_best.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
    
    def train(self, X, y, epochs=200, batch_size=32, n_splits=5):
        """
        Train model with cross-validation and comprehensive evaluation
        
        Args:
            X: Features
            y: Target
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            n_splits: Number of cross-validation folds
        """
        # Create and fit preprocessor
        self.preprocessor = self.create_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Convert target to numeric if needed
        if y.dtype == object:
            y = y.map({'YES': 1, 'NO': 0})
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store metrics for each fold
        fold_metrics = {
            'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
        
        # Cross-validation training
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y), 1):
            logger.info(f"Training fold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X_processed[train_idx], X_processed[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = self.create_model(input_dim=X_processed.shape[1])
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.get_callbacks(),
                verbose=1
            )
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            fold_metrics['auc'].append(roc_auc_score(y_val, y_pred))
            fold_metrics['accuracy'].append(accuracy_score(y_val, y_pred_binary))
            fold_metrics['precision'].append(precision_score(y_val, y_pred_binary))
            fold_metrics['recall'].append(recall_score(y_val, y_pred_binary))
            fold_metrics['f1'].append(f1_score(y_val, y_pred_binary))
            
            # Save best model
            if fold == 1 or roc_auc_score(y_val, y_pred) > max(fold_metrics['auc'][:-1]):
                self.model = model
                self.history = history.history
        
        # Calculate average metrics
        self.metrics = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in fold_metrics.items()
        }
        
        # Save final model and metrics
        self.save_model()
        self.save_metrics()
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)
    
    def save_model(self):
        """Save model and preprocessor"""
        if self.model is not None:
            self.model.save(f'{self.model_path}.h5')
            joblib.dump(self.preprocessor, f'{self.model_path}_preprocessor.pkl')
            logger.info(f"Model saved to {self.model_path}.h5")
    
    def save_metrics(self):
        """Save training metrics"""
        metrics_path = f'{self.model_path}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
    
    @classmethod
    def load(cls, model_path):
        """Load trained model"""
        instance = cls(0)  # input_dim will be set from loaded model
        instance.model = load_model(f'{model_path}.h5')
        instance.preprocessor = joblib.load(f'{model_path}_preprocessor.pkl')
        instance.input_dim = instance.model.layers[0].input_shape[1]
        
        # Load metrics if available
        metrics_path = f'{model_path}_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                instance.metrics = json.load(f)
        
        return instance

def main():
    """Main training function"""
    try:
        # Load and prepare data
        logger.info("Loading data...")
        df = pd.read_csv('data/autism_screening_dataset.csv')
        
        # Preprocess data
        X = df.drop(columns=['Autism_Diagnosis'])
        y = df['Autism_Diagnosis']
        
        # Initialize and train model
        logger.info("Initializing Advanced ANN model...")
        model = AdvancedANNModel(input_dim=X.shape[1])
        
        logger.info("Starting training...")
        metrics = model.train(X, y)
        
        # Log final metrics
        logger.info("\nTraining completed. Final metrics:")
        for metric, values in metrics.items():
            logger.info(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 