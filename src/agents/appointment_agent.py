"""
Appointment Agent - Handles appointment booking and scheduling
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

class AppointmentAgent:
    def __init__(self, llm, appointment_tools):
        self.llm = llm
        self.tools = appointment_tools
    
    def execute(self, plan: Dict, patient_context: Optional[Dict] = None) -> Dict:
        """Execute appointment-related tasks"""
        try:
            tasks = plan.get("tasks", [])
            appointment_tasks = [t for t in tasks if "appointment" in t.lower() or "book" in t.lower()]
            
            if not appointment_tasks:
                return {"status": "no_appointment_tasks", "message": "No appointment tasks found"}
            
            # Extract appointment requirements
            specialty = self._extract_specialty(appointment_tasks[0])
            patient_id = plan.get("patient_id", "unknown")
            
            # Check doctor availability
            available_slots = self.tools.get_available_slots(specialty)
            
            if not available_slots:
                return {
                    "status": "no_availability",
                    "message": f"No available slots for {specialty}",
                    "specialty": specialty
                }
            
            # Book the first available slot
            booking_result = self.tools.book_appointment(
                patient_id=patient_id,
                doctor_id=available_slots[0]["doctor_id"],
                slot_time=available_slots[0]["time"],
                specialty=specialty
            )
            
            return {
                "status": "success",
                "booking": booking_result,
                "available_slots": available_slots[:3],  # Show top 3 options
                "message": f"Appointment booked successfully for {specialty}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Appointment booking failed: {str(e)}"
            }
    
    def _extract_specialty(self, task: str) -> str:
        """Extract medical specialty from task description"""
        specialties = {
            "nephrologist": "nephrology",
            "cardiologist": "cardiology", 
            "neurologist": "neurology",
            "oncologist": "oncology",
            "dermatologist": "dermatology",
            "orthopedic": "orthopedics",
            "pediatrician": "pediatrics",
            "psychiatrist": "psychiatry"
        }
        
        task_lower = task.lower()
        for specialty, field in specialties.items():
            if specialty in task_lower:
                return field
        
        return "general_medicine"
