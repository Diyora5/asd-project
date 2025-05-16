import unittest
from app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AutiScreen', response.data)

    def test_api_predict_missing_data(self):
        response = self.app.post('/api/predict', json={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'success', response.data)

    def test_survey_route(self):
        response = self.app.get('/survey/1')
        self.assertIn(response.status_code, [200, 302])  # 302 if redirect

if __name__ == '__main__':
    unittest.main()