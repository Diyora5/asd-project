from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asd_screening.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

class ASDScreeningApp:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_models()
        
    def load_models(self):
        """Load the trained model and preprocessor"""
        try:
            model_path = os.path.join('models', 'best_model.h5')
            preprocessor_path = os.path.join('models', 'preprocessor.joblib')
            
            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                self.model = tf.keras.models.load_model(model_path)
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Successfully loaded model and preprocessor")
            else:
                logger.error("Model or preprocessor files not found")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def preprocess_answers(self, answers):
        """Preprocess survey answers for model prediction"""
        try:
            # Create a DataFrame with all required columns and proper dtypes
            required_columns = {
                'A1_Score': int, 'A2_Score': int, 'A3_Score': int, 'A4_Score': int, 'A5_Score': int,
                'A6_Score': int, 'A7_Score': int, 'A8_Score': int, 'A9_Score': int, 'A10_Score': int,
                'A11_Score': int, 'A12_Score': int, 'A13_Score': int, 'A14_Score': int, 'A15_Score': int,
                'A16_Score': int, 'A17_Score': int, 'A18_Score': int, 'A19_Score': int, 'A20_Score': int,
                'A21_Score': int, 'A22_Score': int, 'A23_Score': int, 'A24_Score': int, 'A25_Score': int,
                'A26_Score': int, 'A27_Score': int, 'A28_Score': int, 'A29_Score': int, 'A30_Score': int,
                'A31_Score': int, 'A32_Score': int, 'A33_Score': int, 'A34_Score': int, 'A35_Score': int,
                'A36_Score': int, 'A37_Score': int, 'A38_Score': int, 'A39_Score': int,
                'Who_Completed_Test': str, 'Country_of_Residence': str, 'Age': int, 'Gender': str,
                'A30_Score.1': int, 'Sample_ID': str, 'QChat-40-Score': int
            }
            
            # Initialize DataFrame with proper dtypes
            df = pd.DataFrame(columns=required_columns.keys())
            for col, dtype in required_columns.items():
                df[col] = pd.Series(dtype=dtype)
            
            # Fill in the answers from the survey
            for i, answer in enumerate(answers.values(), 1):
                if i <= 39:  # Ensure we don't exceed the number of questions
                    column_name = f'A{i}_Score'
                    df.loc[0, column_name] = int(answer)
            
            # Fill remaining columns with default values
            df.loc[0, 'Who_Completed_Test'] = 'Parent'
            df.loc[0, 'Country_of_Residence'] = 'Unknown'
            df.loc[0, 'Age'] = 2  # Default age
            df.loc[0, 'Gender'] = 'Unknown'
            df.loc[0, 'A30_Score.1'] = 0
            df.loc[0, 'Sample_ID'] = 'SURVEY_' + datetime.now().strftime('%Y%m%d_%H%M%S')
            df.loc[0, 'QChat-40-Score'] = sum(int(v) for v in answers.values())
            
            # Convert all columns to their proper types
            for col, dtype in required_columns.items():
                df[col] = df[col].astype(dtype)
            
            # Apply preprocessing
            if self.preprocessor:
                processed_data = self.preprocessor.transform(df)
                logger.info("Successfully preprocessed answers")
                return processed_data
            return None
        except Exception as e:
            logger.error(f"Error preprocessing answers: {str(e)}")
            return None

    def predict(self, processed_data):
        """Make prediction using the loaded model"""
        try:
            if self.model and processed_data is not None:
                prediction = self.model.predict(processed_data)
                probability = prediction[0][0]
                result = "Positive" if probability >= 0.5 else "Negative"
                confidence = probability if result == "Positive" else 1 - probability
                return result, confidence
            return None, None
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None, None

asd_app = ASDScreeningApp()

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/survey')
def survey():
    return render_template('survey.html')

@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    try:
        answers = request.json
        processed_data = asd_app.preprocess_answers(answers)
        result, confidence = asd_app.predict(processed_data)
        
        if result and confidence:
            return jsonify({
                'result': result,
                'confidence': float(confidence),
                'message': f"The screening result is {result} with {confidence:.2%} confidence."
            })
        return jsonify({'error': 'Failed to process survey'}), 500
    except Exception as e:
        logger.error(f"Error processing survey: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Disable the future warning about downcasting
    pd.set_option('future.no_silent_downcasting', True)
    app.run(debug=True, use_reloader=False) 