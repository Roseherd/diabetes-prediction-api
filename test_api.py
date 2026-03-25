#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Diabetes Prediction API
Tests all endpoints with sample data
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 5

class APITester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        try:
            print("\n[TEST 1] Health Check")
            print("-" * 50)
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            
            if response.status_code == 200:
                print("[OK] Health check passed")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"[FAIL] Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint"""
        try:
            print("\n[TEST 2] Root Endpoint")
            print("-" * 50)
            response = self.session.get(f"{self.base_url}/", timeout=TIMEOUT)
            
            if response.status_code == 200:
                print("[OK] Root endpoint accessible")
                data = response.json()
                print(f"Available endpoints: {list(data.get('endpoints', {}).keys())}")
                return True
            else:
                print(f"[FAIL] Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        try:
            print("\n[TEST 3] Get Model Information")
            print("-" * 50)
            response = self.session.get(f"{self.base_url}/model-info", timeout=TIMEOUT)
            
            if response.status_code == 200:
                print("[OK] Model info retrieved")
                data = response.json()
                print(f"Model Type: {data.get('model_type')}")
                print(f"Number of Features: {data.get('n_features')}")
                print(f"Model Accuracy: {data.get('accuracy'):.4f}")
                print(f"Model ROC-AUC: {data.get('roc_auc'):.4f}")
                print(f"\nFeatures: {', '.join(data.get('feature_names', []))}")
                return True
            else:
                print(f"[FAIL] Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def test_single_prediction(self) -> bool:
        """Test single prediction endpoint"""
        try:
            print("\n[TEST 4] Single Prediction")
            print("-" * 50)
            
            test_data = {
                "gender": "Male",
                "age": 45.0,
                "hypertension": 0,
                "heart_disease": 0,
                "smoking_history": "never",
                "bmi": 25.5,
                "HbA1c_level": 5.8,
                "blood_glucose_level": 120
            }
            
            print("Input Data:")
            print(json.dumps(test_data, indent=2))
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                print("\n[OK] Prediction successful")
                data = response.json()
                prediction = data.get('prediction')
                probability = data.get('probability')
                risk_level = data.get('risk_level')
                
                prediction_text = "HAS DIABETES" if prediction == 1 else "NO DIABETES"
                print(f"Prediction: {prediction_text}")
                print(f"Probability: {probability:.2%}")
                print(f"Risk Level: {risk_level}")
                return True
            else:
                print(f"[FAIL] Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def test_high_risk_prediction(self) -> bool:
        """Test a high-risk patient"""
        try:
            print("\n[TEST 5] High-Risk Patient Prediction")
            print("-" * 50)
            
            test_data = {
                "gender": "Female",
                "age": 65.0,
                "hypertension": 1,
                "heart_disease": 1,
                "smoking_history": "former",
                "bmi": 31.5,
                "HbA1c_level": 7.5,
                "blood_glucose_level": 200
            }
            
            print("Input Data (High-Risk Profile):")
            print(json.dumps(test_data, indent=2))
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                print("\n[OK] Prediction successful")
                data = response.json()
                prediction = data.get('prediction')
                probability = data.get('probability')
                risk_level = data.get('risk_level')
                
                prediction_text = "HAS DIABETES" if prediction == 1 else "NO DIABETES"
                print(f"Prediction: {prediction_text}")
                print(f"Probability: {probability:.2%}")
                print(f"Risk Level: {risk_level}")
                return True
            else:
                print(f"[FAIL] Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        try:
            print("\n[TEST 6] Batch Predictions")
            print("-" * 50)
            
            test_data = {
                "predictions": [
                    {
                        "gender": "Male",
                        "age": 45.0,
                        "hypertension": 0,
                        "heart_disease": 0,
                        "smoking_history": "never",
                        "bmi": 25.5,
                        "HbA1c_level": 5.8,
                        "blood_glucose_level": 120
                    },
                    {
                        "gender": "Female",
                        "age": 55.0,
                        "hypertension": 1,
                        "heart_disease": 0,
                        "smoking_history": "former",
                        "bmi": 28.2,
                        "HbA1c_level": 6.5,
                        "blood_glucose_level": 150
                    },
                    {
                        "gender": "Male",
                        "age": 70.0,
                        "hypertension": 1,
                        "heart_disease": 1,
                        "smoking_history": "current",
                        "bmi": 30.0,
                        "HbA1c_level": 8.0,
                        "blood_glucose_level": 250
                    }
                ]
            }
            
            print(f"Sending batch with {len(test_data['predictions'])} patients...")
            
            response = self.session.post(
                f"{self.base_url}/batch-predict",
                json=test_data,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                print("\n[OK] Batch prediction successful")
                data = response.json()
                total = data.get('total_predictions')
                
                print(f"Total Predictions: {total}")
                print("\nResults:")
                for idx, pred in enumerate(data.get('predictions', []), 1):
                    prediction = pred.get('prediction')
                    probability = pred.get('probability')
                    risk_level = pred.get('risk_level')
                    prediction_text = "HAS DIABETES" if prediction == 1 else "NO DIABETES"
                    print(f"  Patient {idx}: {prediction_text} (Prob: {probability:.2%}, Risk: {risk_level})")
                
                return True
            else:
                print(f"[FAIL] Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def test_invalid_input(self) -> bool:
        """Test error handling with invalid input"""
        try:
            print("\n[TEST 7] Error Handling (Invalid Input)")
            print("-" * 50)
            
            test_data = {
                "gender": "Male",
                "age": 150.0,  # Invalid: age > 120
                "hypertension": 0,
                "heart_disease": 0,
                "smoking_history": "never",
                "bmi": 25.5,
                "HbA1c_level": 5.8,
                "blood_glucose_level": 120
            }
            
            print("Sending invalid age value (150)...")
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                timeout=TIMEOUT
            )
            
            if response.status_code != 200:
                print(f"[OK] API correctly rejected invalid input (Status: {response.status_code})")
                print(f"Error: {response.json().get('detail', 'Unknown error')}")
                return True
            else:
                print("[FAIL] API accepted invalid input")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def run_all_tests(self) -> None:
        """Run all tests and display summary"""
        print("=" * 80)
        print("DIABETES PREDICTION API - TEST SUITE")
        print("=" * 80)
        print(f"API Base URL: {self.base_url}")
        print(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Root Endpoint", self.test_root_endpoint),
            ("Model Info", self.test_model_info),
            ("Single Prediction", self.test_single_prediction),
            ("High-Risk Prediction", self.test_high_risk_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Error Handling", self.test_invalid_input),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                print(f"\n[CRITICAL ERROR] in {test_name}: {e}")
                results[test_name] = False
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "[PASSED]" if result else "[FAILED]"
            print(f"{status} {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n[SUCCESS] All tests passed!")
        else:
            print(f"\n[WARNING] {total - passed} test(s) failed")
        
        print("=" * 80)

if __name__ == "__main__":
    tester = APITester()
    tester.run_all_tests()
