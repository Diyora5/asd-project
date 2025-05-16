from flask import request, jsonify
import logging

# Assuming survey_manager is already imported and initialized in app.py

def register_api_predict(app, survey_manager, logger):
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
