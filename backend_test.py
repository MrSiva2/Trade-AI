#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime
import time

class TradingAITester:
    def __init__(self, base_url="https://trading-model-hub.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict) and len(str(response_data)) < 500:
                        print(f"   Response: {response_data}")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text[:200]}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test basic health endpoint"""
        return self.run_test("Health Check", "GET", "health", 200)

    def test_dashboard_stats(self):
        """Test dashboard stats endpoint"""
        return self.run_test("Dashboard Stats", "GET", "dashboard/stats", 200)

    def test_prebuilt_models(self):
        """Test prebuilt models endpoint"""
        success, data = self.run_test("Prebuilt Models", "GET", "models/prebuilt", 200)
        if success and data.get('models'):
            print(f"   Found {len(data['models'])} prebuilt models")
            return True, data
        return success, data

    def test_data_files(self):
        """Test data files endpoint"""
        success, data = self.run_test("Data Files", "GET", "data/files", 200)
        if success and data.get('files'):
            print(f"   Found {len(data['files'])} CSV files")
            return True, data
        return success, data

    def test_data_preview(self):
        """Test CSV preview functionality"""
        # First get available files
        success, files_data = self.test_data_files()
        if not success or not files_data.get('files'):
            print("❌ No files available for preview test")
            return False, {}
        
        # Try to preview the first file
        first_file = files_data['files'][0]
        file_path = first_file['path']
        
        return self.run_test(
            "CSV Preview", 
            "GET", 
            "data/preview", 
            200, 
            params={"file_path": file_path, "rows": 10}
        )

    def test_training_start(self):
        """Test training start functionality"""
        # Get models and files first
        models_success, models_data = self.test_prebuilt_models()
        files_success, files_data = self.test_data_files()
        
        if not models_success or not files_success:
            print("❌ Cannot test training - missing models or files")
            return False, {}
        
        if not models_data.get('models') or not files_data.get('files'):
            print("❌ Cannot test training - no models or files available")
            return False, {}

        # Get sample data to determine columns
        first_file = files_data['files'][0]
        preview_success, preview_data = self.run_test(
            "Get Columns for Training", 
            "GET", 
            "data/preview", 
            200, 
            params={"file_path": first_file['path'], "rows": 5}
        )
        
        if not preview_success or not preview_data.get('columns'):
            print("❌ Cannot get columns for training test")
            return False, {}

        columns = preview_data['columns']
        if len(columns) < 2:
            print("❌ Not enough columns for training test")
            return False, {}

        # Use first model and first file
        model_id = models_data['models'][0]['id']
        train_data_path = first_file['path']
        target_column = columns[0]  # Use first column as target
        feature_columns = columns[1:3] if len(columns) > 2 else columns[1:2]  # Use next columns as features

        training_config = {
            "model_id": model_id,
            "train_data_path": train_data_path,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "epochs": 5,  # Small number for testing
            "batch_size": 32,
            "validation_split": 0.2
        }

        success, data = self.run_test("Start Training", "POST", "training/start", 200, training_config)
        if success and data.get('session_id'):
            self.session_id = data['session_id']
            print(f"   Training session started: {self.session_id}")
        return success, data

    def test_training_status(self):
        """Test training status endpoint"""
        if not self.session_id:
            print("❌ No training session to check status")
            return False, {}
        
        return self.run_test("Training Status", "GET", f"training/status/{self.session_id}", 200)

    def test_training_logs(self):
        """Test training logs endpoint"""
        if not self.session_id:
            print("❌ No training session to get logs")
            return False, {}
        
        return self.run_test("Training Logs", "GET", f"training/logs/{self.session_id}", 200)

    def test_training_output(self):
        """Test training output endpoint"""
        if not self.session_id:
            print("❌ No training session to get output")
            return False, {}
        
        return self.run_test("Training Output", "GET", f"training/output/{self.session_id}", 200)

    def test_all_sessions(self):
        """Test get all training sessions"""
        return self.run_test("All Training Sessions", "GET", "training/sessions", 200)

    def test_custom_models(self):
        """Test custom models endpoint"""
        return self.run_test("Custom Models", "GET", "models/custom", 200)

    def test_saved_models(self):
        """Test saved models endpoint"""
        return self.run_test("Saved Models", "GET", "models/saved", 200)

    def test_backtest_results(self):
        """Test backtest results endpoint"""
        return self.run_test("Backtest Results", "GET", "backtest/results", 200)

    def test_data_folders(self):
        """Test data folders endpoint"""
        return self.run_test("Data Folders", "GET", "data/folders", 200)

def main():
    print("🚀 Starting Trading AI Model Hub Backend Tests")
    print("=" * 60)
    
    tester = TradingAITester()
    
    # Basic connectivity tests
    print("\n📡 CONNECTIVITY TESTS")
    tester.test_health_check()
    
    # Dashboard tests
    print("\n📊 DASHBOARD TESTS")
    tester.test_dashboard_stats()
    
    # Data management tests
    print("\n📁 DATA MANAGEMENT TESTS")
    tester.test_data_folders()
    tester.test_data_files()
    tester.test_data_preview()
    
    # Model management tests
    print("\n🧠 MODEL MANAGEMENT TESTS")
    tester.test_prebuilt_models()
    tester.test_custom_models()
    tester.test_saved_models()
    
    # Training tests
    print("\n🏋️ TRAINING TESTS")
    tester.test_training_start()
    
    # Wait a bit for training to start
    if tester.session_id:
        print("\n⏳ Waiting for training to start...")
        time.sleep(3)
        tester.test_training_status()
        tester.test_training_logs()
        tester.test_training_output()
    
    tester.test_all_sessions()
    
    # Backtesting tests
    print("\n📈 BACKTESTING TESTS")
    tester.test_backtest_results()
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS")
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {tester.tests_run - tester.tests_passed}")
    print(f"Success Rate: {(tester.tests_passed / tester.tests_run * 100):.1f}%")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())