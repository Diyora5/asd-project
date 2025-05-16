from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Change this in production

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('survey_app.log')
    ]
)
logger = logging.getLogger(__name__)

class SurveyManager:
    def __init__(self):
        self.questions = self.load_questions()
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()
        self.part_size = 10  # 40 questions divided into 4 parts
        
    def load_questions(self):
        """Load survey questions from file"""
        questions = []
        try:
            with open('data/screening_questionnaire.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        questions.append(line.strip())
            logger.info(f"Loaded {len(questions)} questions")
            return questions
        except Exception as e:
            logger.error(f"Error loading questions: {str(e)}")
            return []
    
    def load_model(self):
        """Load the best performing model"""
        try:
            model_path = 'models/advanced_ann_best.h5'
            if os.path.exists(model_path):
                return load_model(model_path)
            else:
                logger.error("Model file not found")
                return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def load_preprocessor(self):
        """Load the preprocessor"""
        try:
            preprocessor_path = 'models/advanced_ann_preprocessor.pkl'
            if os.path.exists(preprocessor_path):
                preprocessor = joblib.load(preprocessor_path)
                logger.info("Successfully loaded preprocessor")
                return preprocessor
            else:
                logger.error(f"Preprocessor file not found at {preprocessor_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            return None
    
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
                # Convert to numpy array and ensure correct shape
                processed_data = np.array(processed_data).reshape(1, -1)
                probability = self.model.predict(processed_data)[0][0]
                logger.info(f"Prediction made with probability: {probability}")
                return float(probability)
            return None
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def get_interpretation(self, probability):
        """Get interpretation of the prediction"""
        if probability is None:
            return {
                'level': 'Unknown',
                'message': 'Unable to make prediction. Please try again.',
                'color': 'secondary'
            }
        
        if probability < 0.3:
            return {
                'level': 'Low',
                'message': 'Based on the responses, there are minimal signs of ASD. However, if you have concerns, please consult a healthcare professional.',
                'color': 'success'
            }
        elif probability < 0.6:
            return {
                'level': 'Moderate',
                'message': 'Some signs of ASD are present. It is recommended to consult a healthcare professional for further evaluation.',
                'color': 'warning'
            }
        else:
            return {
                'level': 'High',
                'message': 'Significant signs of ASD are present. It is strongly recommended to consult a healthcare professional for a comprehensive evaluation.',
                'color': 'danger'
            }
    
    def get_part_questions(self, part_number):
        """Get questions for a specific part"""
        start_idx = (part_number - 1) * self.part_size
        end_idx = start_idx + self.part_size
        return self.questions[start_idx:end_idx]

survey_manager = SurveyManager()

from flask import send_from_directory
import os

@app.route('/')
def index():
    # Correct the filename extension typo if any
    filename = 'asd_project_mobile_app.html'
    # Check if file exists in root directory
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    # Fallback to static folder
    return send_from_directory('static', filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        answers = data.get('answers', {})
        # Preprocess answers
        processed_data = survey_manager.preprocess_answers(answers)
        # Make prediction
        probability = survey_manager.predict(processed_data)
        # Get interpretation
        interpretation = survey_manager.get_interpretation(probability)
        return jsonify({
            'success': True,
            'probability': probability,
            'interpretation': interpretation
        })
    except Exception as e:
        logger.error(f"Error in /api/predict: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/survey/<int:part>')
def survey(part):
    """Render the survey page for a specific part"""
    if part < 1 or part > 4:
        return redirect(url_for('welcome'))
    
    questions = survey_manager.get_part_questions(part)
    return render_template('survey.html', 
                         questions=questions,
                         part=part,
                         total_parts=4)

@app.route('/submit_part', methods=['POST'])
def submit_part():
    """Handle part submission"""
    try:
        part = int(request.form.get('part'))
        answers = {k: v for k, v in request.form.items() if k.startswith('q')}
        
        # Store answers in session
        if 'answers' not in session:
            session['answers'] = {}
        session['answers'].update(answers)
        
        # If this is the last part, process all answers
        if part == 4:
            return process_complete_survey()
        
        return jsonify({
            'success': True,
            'next_part': part + 1
        })
        
    except Exception as e:
        logger.error(f"Error processing part submission: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def process_complete_survey():
    """Process complete survey and return results"""
    try:
        answers = session.get('answers', {})
        
        # Preprocess answers
        processed_data = survey_manager.preprocess_answers(answers)
        
        # Make prediction
        probability = survey_manager.predict(processed_data)
        
        # Get interpretation
        interpretation = survey_manager.get_interpretation(probability)
        
        # Save results
        save_results(answers, probability, interpretation)
        
        # Store results in session for display
        session['probability'] = probability
        session['interpretation'] = interpretation
        
        # Clear answers from session
        session.pop('answers', None)
        
        return jsonify({
            'success': True,
            'redirect': url_for('results')
        })
        
    except Exception as e:
        logger.error(f"Error processing complete survey: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def save_results(answers, probability, interpretation):
    """Save survey results"""
    try:
        results_dir = 'survey_results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{results_dir}/survey_result_{timestamp}.json'
        
        result = {
            'timestamp': timestamp,
            'answers': answers,
            'probability': probability,
            'interpretation': interpretation
        }
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
            
        logger.info(f"Saved survey results to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving survey results: {str(e)}")

@app.route('/results')
def results():
    """Render the results page"""
    # Get the last processed survey results from session
    probability = session.get('probability')
    interpretation = session.get('interpretation')
    
    if not probability or not interpretation:
        return redirect(url_for('welcome'))
    
    return render_template('results.html', 
                         probability=probability,
                         interpretation=interpretation)

@app.route('/result')
def show_result():
    result = request.args.get('result', 'Negative')
    score = float(request.args.get('score', 0))
    return render_template('result.html', result=result, score=score)

if __name__ == '__main__':
    # Disable the future warning about downcasting
    pd.set_option('future.no_silent_downcasting', True)
    app.run(debug=True, port=5000)
