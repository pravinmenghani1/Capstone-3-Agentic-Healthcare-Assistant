"""
Medical Records Management Tools
"""

import json
import os
import pandas as pd
import PyPDF2
from typing import Dict, List, Any, Optional
from datetime import datetime

class MedicalTools:
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        
        # Load patient records from Excel
        self.patient_records = self._load_patient_records()
        
        # Load PDF reports
        self.pdf_reports = self._load_pdf_reports()
    
    def _load_patient_records(self) -> Dict:
        """Load patient records from Excel file"""
        try:
            excel_file = os.path.join(self.data_dir, "records.xlsx")
            if os.path.exists(excel_file):
                df = pd.read_excel(excel_file)
                
                # Convert to dictionary format using actual Excel structure
                records = {}
                for _, row in df.iterrows():
                    name = str(row.get('Name', 'Unknown')).strip()
                    if name and name != 'Unknown':
                        # Use first name as patient ID (lowercase)
                        patient_id = name.split()[0].lower()
                        
                        # Extract conditions and medications from summary
                        summary = str(row.get('Summary', ''))
                        conditions, medications = self._extract_medical_info_from_summary(summary)
                        
                        records[patient_id] = {
                            "name": name,
                            "age": int(row.get('Age', 0)) if pd.notna(row.get('Age')) else 0,
                            "gender": str(row.get('Gender', 'Unknown')),
                            "phone": str(row.get('Phone_number', 'Unknown')),
                            "address": str(row.get('Address', 'Unknown')),
                            "summary": summary,
                            "conditions": conditions,
                            "medications": medications,
                            "last_visit": "2024-03-01"  # Default recent date
                        }
                
                return records
            
            return {}
            
        except Exception as e:
            print(f"Error loading patient records: {e}")
            return {}
    
    def _extract_medical_info_from_summary(self, summary: str) -> tuple:
        """Extract conditions and medications from patient summary"""
        conditions = []
        medications = []
        
        if pd.isna(summary) or summary == 'nan':
            return conditions, medications
        
        summary_lower = summary.lower()
        
        # Extract conditions based on common medical terms
        condition_keywords = {
            "hypertension": ["hypertension", "high blood pressure", "bp"],
            "diabetes": ["diabetes", "diabetic", "blood sugar"],
            "respiratory infection": ["respiratory infection", "cough", "fever"],
            "upper respiratory infection": ["upper respiratory", "uri"],
            "type 2 diabetes": ["type 2 diabetes", "t2dm"]
        }
        
        for condition, keywords in condition_keywords.items():
            if any(keyword in summary_lower for keyword in keywords):
                conditions.append(condition.title())
        
        # Extract medications
        medication_keywords = [
            "metformin", "antihistamine", "medication", "medicine"
        ]
        
        for med_keyword in medication_keywords:
            if med_keyword in summary_lower:
                medications.append(med_keyword.title())
        
        # If no specific conditions found but summary exists, add general condition
        if not conditions and summary and summary != 'nan':
            if "routine" in summary_lower:
                conditions.append("Routine Checkup")
            else:
                conditions.append("Under Medical Care")
        
        return conditions, medications
    
    def _create_mock_records(self) -> Dict:
        """Create mock patient records"""
        return {
            "ramesh": {
                "name": "Ramesh Kulkarni",
                "age": 65,
                "gender": "Male",
                "phone": "+91-98220-45322",
                "conditions": ["Essential Hypertension", "Diabetes Type 2"],
                "medications": ["Telmisartan", "Metformin"],
                "last_visit": "2024-02-28",
                "address": "52 Residency Road, Chennai"
            },
            "anjali": {
                "name": "Anjali Sharma",
                "age": 42,
                "gender": "Female", 
                "phone": "+91-98765-43210",
                "conditions": ["Migraine", "Anxiety"],
                "medications": ["Sumatriptan", "Sertraline"],
                "last_visit": "2024-03-15"
            },
            "david": {
                "name": "David Wilson",
                "age": 58,
                "gender": "Male",
                "phone": "+91-99887-76543",
                "conditions": ["Chronic Kidney Disease", "Hypertension"],
                "medications": ["Lisinopril", "Furosemide"],
                "last_visit": "2024-03-10"
            },
            "father": {
                "name": "Patient's Father",
                "age": 70,
                "gender": "Male",
                "phone": "Unknown",
                "conditions": ["Chronic Kidney Disease", "Diabetes"],
                "medications": ["ACE Inhibitor", "Insulin"],
                "last_visit": "2024-03-01"
            }
        }
    
    def _load_pdf_reports(self) -> Dict:
        """Load and extract text from PDF reports"""
        reports = {}
        
        try:
            pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
            
            for pdf_file in pdf_files:
                try:
                    file_path = os.path.join(self.data_dir, pdf_file)
                    
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        
                        for page in reader.pages:
                            text += page.extract_text()
                        
                        # Extract patient name from filename or content
                        patient_name = pdf_file.replace('sample_report_', '').replace('.pdf', '')
                        reports[patient_name] = {
                            "filename": pdf_file,
                            "content": text,
                            "extracted_at": datetime.now().isoformat()
                        }
                        
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")
            
        except Exception as e:
            print(f"Error loading PDF reports: {e}")
        
        return reports
    
    def get_patient_history(self, patient_id: str) -> Optional[Dict]:
        """Get comprehensive patient medical history"""
        try:
            # Normalize patient ID
            patient_id = patient_id.lower().strip()
            
            # Check direct match first
            if patient_id in self.patient_records:
                patient_data = self.patient_records[patient_id].copy()
            else:
                # Try to find by name matching
                patient_data = None
                for pid, data in self.patient_records.items():
                    if (patient_id in data.get("name", "").lower() or 
                        patient_id in pid.lower()):
                        patient_data = data.copy()
                        break
                
                if not patient_data:
                    return None
            
            # Add PDF report if available
            if patient_id in self.pdf_reports:
                patient_data["medical_report"] = self.pdf_reports[patient_id]
            
            # Add visit history (mock data)
            patient_data["visit_history"] = self._get_visit_history(patient_id)
            
            # Add lab results (mock data)
            patient_data["lab_results"] = self._get_lab_results(patient_id)
            
            return patient_data
            
        except Exception as e:
            print(f"Error getting patient history: {e}")
            return None
    
    def _get_visit_history(self, patient_id: str) -> List[Dict]:
        """Get mock visit history"""
        base_visits = [
            {
                "date": "2024-03-01",
                "type": "Regular Checkup",
                "doctor": "Dr. Smith",
                "diagnosis": "Routine follow-up",
                "notes": "Patient stable, continue current medications"
            },
            {
                "date": "2024-02-15", 
                "type": "Lab Review",
                "doctor": "Dr. Johnson",
                "diagnosis": "Lab results review",
                "notes": "Blood pressure well controlled"
            }
        ]
        
        return base_visits
    
    def _get_lab_results(self, patient_id: str) -> List[Dict]:
        """Get mock lab results"""
        base_labs = [
            {
                "date": "2024-02-28",
                "test": "Complete Blood Count",
                "results": {"WBC": "7.2", "RBC": "4.5", "Hemoglobin": "13.8"},
                "status": "Normal"
            },
            {
                "date": "2024-02-28",
                "test": "Basic Metabolic Panel", 
                "results": {"Glucose": "110", "Creatinine": "1.1", "BUN": "18"},
                "status": "Normal"
            }
        ]
        
        return base_labs
    
    def add_patient_record(self, patient_data: Dict) -> Dict:
        """Add new patient record"""
        try:
            patient_id = patient_data.get("patient_id", "").lower()
            
            if not patient_id:
                return {"success": False, "message": "Patient ID required"}
            
            # Add timestamp
            patient_data["created_at"] = datetime.now().isoformat()
            
            # Store record
            self.patient_records[patient_id] = patient_data
            
            # Save to file (in production, save to database)
            self._save_patient_records()
            
            return {
                "success": True,
                "message": f"Patient record added for {patient_id}",
                "patient_id": patient_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to add patient record: {str(e)}"
            }
    
    def update_patient_record(self, patient_id: str, updates: Dict) -> Dict:
        """Update existing patient record"""
        try:
            patient_id = patient_id.lower()
            
            if patient_id not in self.patient_records:
                return {"success": False, "message": "Patient not found"}
            
            # Update record
            self.patient_records[patient_id].update(updates)
            self.patient_records[patient_id]["updated_at"] = datetime.now().isoformat()
            
            # Save changes
            self._save_patient_records()
            
            return {
                "success": True,
                "message": f"Patient record updated for {patient_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to update patient record: {str(e)}"
            }
    
    def _save_patient_records(self):
        """Save patient records to file"""
        try:
            records_file = os.path.join("src/data", "patient_records.json")
            os.makedirs(os.path.dirname(records_file), exist_ok=True)
            
            with open(records_file, 'w') as f:
                json.dump(self.patient_records, f, indent=2)
                
        except Exception as e:
            print(f"Error saving patient records: {e}")
    
    def search_patients(self, query: str) -> List[Dict]:
        """Search patients by name or condition"""
        results = []
        query_lower = query.lower()
        
        for patient_id, data in self.patient_records.items():
            # Search in name
            if query_lower in data.get("name", "").lower():
                results.append({"patient_id": patient_id, **data, "match_type": "name"})
                continue
            
            # Search in conditions
            conditions = data.get("conditions", [])
            if any(query_lower in condition.lower() for condition in conditions):
                results.append({"patient_id": patient_id, **data, "match_type": "condition"})
        
        return results
