import unittest
import requests
import time
from datetime import datetime, timedelta

class TestSalesForecastAPI(unittest.TestCase):
    BASE_URL = "http://localhost:5001"
    
    def setUp(self):
        """Setup test case"""
        # Check if API is running
        try:
            response = requests.get(f"{self.BASE_URL}/status")
            if response.status_code != 200:
                self.fail("API is not running. Please start the API server first.")
        except requests.ConnectionError:
            self.fail("Could not connect to API. Please start the API server first.")
    
    def test_api_status(self):
        """Test /status endpoint"""
        response = requests.get(f"{self.BASE_URL}/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
        self.assertIn('active_jobs', data)
        self.assertIn('jobs_by_status', data)
        
        # Check if model is loaded
        self.assertTrue(data['model_loaded'], "Model should be loaded")
    
    def test_prediction_workflow(self):
        """Test complete prediction workflow"""
        # 1. Submit prediction job
        prediction_data = {
            "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "store": 1,
            "item": 1
        }
        
        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_data
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('job_id', data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'pending')
        
        job_id = data['job_id']
        
        # 2. Check job status until completion
        max_retries = 10
        retry_count = 0
        job_completed = False
        
        while retry_count < max_retries and not job_completed:
            response = requests.get(f"{self.BASE_URL}/status/{job_id}")
            self.assertEqual(response.status_code, 200)
            
            status_data = response.json()
            if status_data['status'] in ['completed', 'failed']:
                job_completed = True
                break
                
            retry_count += 1
            time.sleep(1)  # Wait before next check
        
        self.assertTrue(job_completed, "Job should complete within timeout")
        
        if status_data['status'] == 'completed':
            self.assertIn('result', status_data)
            self.assertIn('predicted_sales', status_data['result'])
    
    def test_invalid_job_status(self):
        """Test status check for non-existent job"""
        response = requests.get(f"{self.BASE_URL}/status/invalid-job-id")
        self.assertEqual(response.status_code, 404)
        
        data = response.json()
        self.assertIn('error', data)
    
    def test_invalid_prediction_request(self):
        """Test prediction with invalid data"""
        test_cases = [
            {
                "desc": "Missing required fields",
                "data": {"date": "2024-03-14"},
                "expected_code": 400,
                "error_contains": "Missing required field"
            },
            {
                "desc": "Invalid date format",
                "data": {
                    "date": "invalid-date",
                    "store": 1,
                    "item": 1
                },
                "expected_code": 400,
                "error_contains": "Invalid date format"
            },
            {
                "desc": "Invalid store value",
                "data": {
                    "date": "2024-03-14",
                    "store": -1,
                    "item": 1
                },
                "expected_code": 400,
                "error_contains": "Store must be a positive integer"
            },
            {
                "desc": "Invalid item value",
                "data": {
                    "date": "2024-03-14",
                    "store": 1,
                    "item": "invalid"
                },
                "expected_code": 400,
                "error_contains": "Item must be a positive integer"
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case["desc"]):
                print(f"\nTesting: {test_case['desc']}")
                print(f"Input data: {test_case['data']}")
                
                response = requests.post(
                    f"{self.BASE_URL}/predict",
                    json=test_case["data"]
                )
                
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.json()}")
                
                self.assertEqual(
                    response.status_code,
                    test_case["expected_code"],
                    f"Failed on: {test_case['desc']}\nExpected {test_case['expected_code']}, got {response.status_code}\nResponse: {response.json()}"
                )
                
                data = response.json()
                self.assertIn('error', data, f"No error message in response: {data}")
                self.assertIn(
                    test_case["error_contains"],
                    data['error'],
                    f"Error message '{data['error']}' doesn't contain '{test_case['error_contains']}'"
                )
    
    def test_jobs_listing(self):
        """Test /jobs endpoint with filters"""
        # Submit a test job first
        prediction_data = {
            "date": "2024-03-14",
            "store": 1,
            "item": 1
        }
        requests.post(f"{self.BASE_URL}/predict", json=prediction_data)
        
        # Test jobs listing
        response = requests.get(f"{self.BASE_URL}/jobs")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('total_jobs', data)
        self.assertIn('filtered_jobs', data)
        self.assertIn('jobs', data)
        
        # Test with status filter
        response = requests.get(f"{self.BASE_URL}/jobs?status=completed")
        self.assertEqual(response.status_code, 200)
        
        # Test with limit
        response = requests.get(f"{self.BASE_URL}/jobs?limit=1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(len(data['jobs']), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2) 