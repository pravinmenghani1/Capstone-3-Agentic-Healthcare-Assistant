"""
Enhanced Appointment Agent with Context Awareness and Proper Prompt Engineering
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

class EnhancedAppointmentAgent:
    def __init__(self, llm, appointment_tools):
        self.llm = llm
        self.tools = appointment_tools
        
        self.appointment_prompt_template = """
You are a healthcare appointment booking specialist. Process the appointment request with full context awareness.

Patient Context: {patient_context}
Appointment Plan: {plan}
Available Slots: {available_slots}

Your task:
1. Analyze patient medical history for specialty matching
2. Consider urgency based on medical conditions
3. Select optimal appointment slot
4. Generate professional booking confirmation

Patient Medical Considerations:
- Review chronic conditions for specialist matching
- Check medication interactions that might affect timing
- Consider patient age and mobility for scheduling
- Identify any urgent medical needs

Provide detailed appointment booking analysis and confirmation.
"""
    
    def execute_with_context(self, plan: Dict, patient_context: Optional[Dict] = None) -> Dict:
        """Execute appointment booking with full context awareness"""
        try:
            # Extract appointment requirements from plan
            appointment_requirements = self._analyze_appointment_requirements(plan, patient_context)
            
            # Get available slots based on requirements
            available_slots = self.tools.get_available_slots(
                appointment_requirements["specialty"],
                days_ahead=appointment_requirements["urgency_days"]
            )
            
            if not available_slots:
                return {
                    "status": "no_availability",
                    "message": f"No available slots for {appointment_requirements['specialty']}",
                    "specialty": appointment_requirements["specialty"],
                    "patient_context_considered": True
                }
            
            # Use LLM to select optimal slot with context
            optimal_slot = self._select_optimal_slot_with_llm(
                available_slots, 
                patient_context, 
                appointment_requirements
            )
            
            # Book the appointment
            booking_result = self.tools.book_appointment(
                patient_id=plan.get("patient_id", "unknown"),
                doctor_id=optimal_slot["doctor_id"],
                slot_time=optimal_slot["time"],
                specialty=appointment_requirements["specialty"]
            )
            
            if booking_result["success"]:
                # Generate comprehensive confirmation with LLM
                confirmation = self._generate_booking_confirmation(
                    booking_result, 
                    patient_context, 
                    appointment_requirements
                )
                
                return {
                    "status": "success",
                    "booking": booking_result,
                    "confirmation": confirmation,
                    "patient_context_used": patient_context is not None,
                    "specialty": appointment_requirements["specialty"],
                    "urgency_level": appointment_requirements["urgency_level"]
                }
            else:
                return {
                    "status": "booking_failed",
                    "message": booking_result.get("message", "Booking failed"),
                    "specialty": appointment_requirements["specialty"]
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Appointment booking failed: {str(e)}",
                "patient_context_available": patient_context is not None
            }
    
    def _analyze_appointment_requirements(self, plan: Dict, patient_context: Optional[Dict]) -> Dict:
        """Analyze appointment requirements from plan and context"""
        requirements = {
            "specialty": "general_medicine",
            "urgency_level": "routine",
            "urgency_days": 14,
            "special_considerations": []
        }
        
        # Extract specialty from plan tasks
        tasks = plan.get("tasks", [])
        for task in tasks:
            task_lower = task.lower()
            if "nephrologist" in task_lower or "kidney" in task_lower:
                requirements["specialty"] = "nephrology"
            elif "cardiologist" in task_lower or "heart" in task_lower:
                requirements["specialty"] = "cardiology"
            elif "neurologist" in task_lower or "brain" in task_lower:
                requirements["specialty"] = "neurology"
            elif "oncologist" in task_lower or "cancer" in task_lower:
                requirements["specialty"] = "oncology"
        
        # Analyze patient context for urgency and considerations
        if patient_context:
            patient_data = patient_context.get("data", {})
            conditions = patient_data.get("conditions", [])
            
            # Check for urgent conditions
            urgent_conditions = ["acute", "severe", "critical", "emergency"]
            if any(urgent in str(conditions).lower() for urgent in urgent_conditions):
                requirements["urgency_level"] = "urgent"
                requirements["urgency_days"] = 3
            
            # Add special considerations
            if "diabetes" in str(conditions).lower():
                requirements["special_considerations"].append("diabetes_management")
            
            if "hypertension" in str(conditions).lower():
                requirements["special_considerations"].append("blood_pressure_monitoring")
            
            # Consider patient age
            age = patient_data.get("age")
            if age and (isinstance(age, int) and age > 65):
                requirements["special_considerations"].append("senior_patient")
        
        # Check plan priority
        if plan.get("priority") == "high":
            requirements["urgency_level"] = "high_priority"
            requirements["urgency_days"] = 7
        
        return requirements
    
    def _select_optimal_slot_with_llm(self, available_slots: List[Dict], 
                                    patient_context: Optional[Dict], 
                                    requirements: Dict) -> Dict:
        """Use LLM to select optimal appointment slot"""
        try:
            # Prepare context for LLM
            context_str = json.dumps(patient_context, indent=2) if patient_context else "No patient context"
            slots_str = json.dumps(available_slots[:5], indent=2)  # Top 5 slots
            
            selection_prompt = f"""
            Select the optimal appointment slot considering patient needs:
            
            Patient Context: {context_str}
            Requirements: {json.dumps(requirements, indent=2)}
            Available Slots: {slots_str}
            
            Consider:
            1. Patient medical conditions and urgency
            2. Doctor specialization match
            3. Optimal timing for patient age/condition
            4. Appointment spacing for chronic conditions
            
            Return the index (0-based) of the best slot and reasoning.
            Format: {{"selected_index": 0, "reasoning": "explanation"}}
            """
            
            response = self.llm.invoke(selection_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract selection
            import re
            index_match = re.search(r'"selected_index":\s*(\d+)', response_content)
            if index_match:
                selected_index = int(index_match.group(1))
                if 0 <= selected_index < len(available_slots):
                    return available_slots[selected_index]
            
            # Fallback to first available slot
            return available_slots[0]
            
        except Exception as e:
            # Fallback to rule-based selection
            return self._rule_based_slot_selection(available_slots, requirements)
    
    def _rule_based_slot_selection(self, available_slots: List[Dict], requirements: Dict) -> Dict:
        """Rule-based slot selection fallback"""
        # Prefer earlier slots for urgent cases
        if requirements["urgency_level"] in ["urgent", "high_priority"]:
            return available_slots[0]
        
        # Prefer mid-morning slots for senior patients
        if "senior_patient" in requirements["special_considerations"]:
            for slot in available_slots:
                slot_time = datetime.fromisoformat(slot["time"].replace('Z', '+00:00'))
                if 9 <= slot_time.hour <= 11:  # 9-11 AM
                    return slot
        
        # Default to first available
        return available_slots[0]
    
    def _generate_booking_confirmation(self, booking_result: Dict, 
                                     patient_context: Optional[Dict], 
                                     requirements: Dict) -> str:
        """Generate comprehensive booking confirmation using LLM"""
        try:
            confirmation_prompt = f"""
            Generate a professional appointment confirmation message:
            
            Booking Details: {json.dumps(booking_result, indent=2)}
            Patient Context: {json.dumps(patient_context, indent=2) if patient_context else "No context"}
            Requirements: {json.dumps(requirements, indent=2)}
            
            Include:
            1. Appointment confirmation details
            2. Doctor information and specialty
            3. Preparation instructions based on patient conditions
            4. Important reminders for the patient
            5. Contact information for changes
            
            Keep it professional, clear, and patient-focused.
            """
            
            response = self.llm.invoke(confirmation_prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            # Fallback confirmation
            appointment = booking_result.get("appointment", {})
            return f"""
            ✅ Appointment Confirmed
            
            Patient: {appointment.get('patient_id', 'Unknown')}
            Doctor: {appointment.get('doctor_name', 'Unknown')}
            Specialty: {requirements['specialty'].title()}
            Date & Time: {appointment.get('appointment_time', 'TBD')}
            
            Please arrive 15 minutes early and bring:
            - Valid ID
            - Insurance card
            - Current medication list
            - Previous medical records if available
            
            For changes or cancellations, please call our office at least 24 hours in advance.
            """
