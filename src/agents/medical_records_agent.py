"""
Medical Records Agent - Manages patient medical history and records
"""

from typing import Dict, List, Any, Optional
import json

class MedicalRecordsAgent:
    def __init__(self, llm, medical_tools):
        self.llm = llm
        self.tools = medical_tools
    
    def execute(self, plan: Dict, patient_context: Optional[Dict] = None) -> Dict:
        """Execute medical records related tasks"""
        try:
            tasks = plan.get("tasks", [])
            records_tasks = [t for t in tasks if any(keyword in t.lower() 
                           for keyword in ["record", "history", "retrieve", "summary"])]
            
            if not records_tasks:
                return {"status": "no_records_tasks", "message": "No medical records tasks found"}
            
            patient_id = plan.get("patient_id", "unknown")
            
            # Retrieve patient medical history
            medical_history = self.tools.get_patient_history(patient_id)
            
            if not medical_history:
                return {
                    "status": "no_records",
                    "message": f"No medical records found for patient {patient_id}",
                    "patient_id": patient_id
                }
            
            # Generate summary using LLM
            summary = self._generate_medical_summary(medical_history, records_tasks[0])
            
            return {
                "status": "success",
                "patient_id": patient_id,
                "medical_history": medical_history,
                "summary": summary,
                "message": "Medical records retrieved and summarized successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Medical records retrieval failed: {str(e)}"
            }
    
    def _generate_medical_summary(self, medical_history: Dict, task: str) -> str:
        """Generate AI summary of medical history"""
        try:
            prompt = f"""
            Summarize the following medical history for healthcare context:
            
            Patient History: {json.dumps(medical_history, indent=2)}
            
            Task Context: {task}
            
            Provide a concise medical summary highlighting:
            1. Key diagnoses and conditions
            2. Current medications
            3. Recent visits and treatments
            4. Relevant alerts or concerns
            
            Format as a professional medical summary.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
