#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Client for Diabetes Prediction API
Easy-to-use client for making predictions
"""

import requests
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class PatientData:
    """Patient data for prediction"""
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gender': self.gender,
            'age': self.age,
            'hypertension': self.hypertension,
            'heart_disease': self.heart_disease,
            'smoking_history': self.smoking_history,
            'bmi': self.bmi,
            'HbA1c_level': self.HbA1c_level,
            'blood_glucose_level': self.blood_glucose_level
        }

class DiabetesPredictionClient:
    """Client for interacting with Diabetes Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Add API key to headers if provided
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
    
    def health_check(self) -> bool:
        """
        Check if API is healthy
        
        Returns:
            bool: True if API is healthy
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metrics
        
        Returns:
            dict: Model information
        """
        response = self.session.get(f"{self.base_url}/model-info", timeout=5)
        response.raise_for_status()
        return response.json()
    
    def predict(self, patient_data: PatientData) -> Dict[str, Any]:
        """
        Make a single prediction
        
        Args:
            patient_data: PatientData object or dict
            
        Returns:
            dict: Prediction result with probability and risk level
        """
        if isinstance(patient_data, PatientData):
            patient_data = patient_data.to_dict()
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=patient_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def batch_predict(self, patients: List[PatientData]) -> Dict[str, Any]:
        """
        Make predictions for multiple patients
        
        Args:
            patients: List of PatientData objects
            
        Returns:
            dict: Batch prediction results
        """
        patient_dicts = [
            p.to_dict() if isinstance(p, PatientData) else p 
            for p in patients
        ]
        
        response = self.session.post(
            f"{self.base_url}/batch-predict",
            json={"predictions": patient_dicts},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def predict_dict(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction from a dictionary
        
        Args:
            patient_dict: Patient data as dictionary
            
        Returns:
            dict: Prediction result
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json=patient_dict,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = DiabetesPredictionClient()
    
    # Check if API is available
    if not client.health_check():
        print("ERROR: API is not available. Make sure it's running on http://localhost:8000")
        exit(1)
    
    print("[OK] API is available\n")
    
    # Get model information
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    model_info = client.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Accuracy: {model_info['accuracy']:.2%}")
    print(f"ROC-AUC: {model_info['roc_auc']:.2%}\n")
    
    # Single prediction
    print("=" * 60)
    print("SINGLE PREDICTION")
    print("=" * 60)
    
    patient = PatientData(
        gender="Male",
        age=45.0,
        hypertension=0,
        heart_disease=0,
        smoking_history="never",
        bmi=25.5,
        HbA1c_level=5.8,
        blood_glucose_level=120
    )
    
    result = client.predict(patient)
    print(f"Patient Data: {patient}")
    print(f"Prediction: {'HAS DIABETES' if result['prediction'] == 1 else 'NO DIABETES'}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}\n")
    
    # Batch prediction
    print("=" * 60)
    print("BATCH PREDICTIONS")
    print("=" * 60)
    
    patients = [
        PatientData("Male", 45.0, 0, 0, "never", 25.5, 5.8, 120),
        PatientData("Female", 55.0, 1, 0, "former", 28.2, 6.5, 150),
        PatientData("Male", 70.0, 1, 1, "current", 30.0, 8.0, 250),
    ]
    
    batch_results = client.batch_predict(patients)
    for i, pred in enumerate(batch_results['predictions'], 1):
        print(f"Patient {i}: {pred['risk_level']} risk - {pred['probability']:.1%} probability")
