"""
Appointment Management Tools
"""

import json
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import random

class AppointmentTools:
    def __init__(self, data_dir: str = "src/data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Mock doctor database
        self.doctors = {
            "nephrology": [
                {"id": "dr_001", "name": "Dr. Sarah Johnson", "specialty": "nephrology"},
                {"id": "dr_002", "name": "Dr. Michael Chen", "specialty": "nephrology"}
            ],
            "cardiology": [
                {"id": "dr_003", "name": "Dr. Emily Davis", "specialty": "cardiology"},
                {"id": "dr_004", "name": "Dr. Robert Wilson", "specialty": "cardiology"}
            ],
            "general_medicine": [
                {"id": "dr_005", "name": "Dr. Lisa Anderson", "specialty": "general_medicine"},
                {"id": "dr_006", "name": "Dr. James Brown", "specialty": "general_medicine"}
            ]
        }
        
        # Load existing appointments
        self.appointments = self._load_appointments()
    
    def _load_appointments(self) -> Dict:
        """Load existing appointments from file"""
        try:
            appointments_file = os.path.join(self.data_dir, "appointments.json")
            if os.path.exists(appointments_file):
                with open(appointments_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def _save_appointments(self):
        """Save appointments to file"""
        try:
            appointments_file = os.path.join(self.data_dir, "appointments.json")
            with open(appointments_file, 'w') as f:
                json.dump(self.appointments, f, indent=2)
        except Exception as e:
            print(f"Error saving appointments: {e}")
    
    def get_available_slots(self, specialty: str, days_ahead: int = 7) -> List[Dict]:
        """Get available appointment slots for a specialty"""
        try:
            if specialty not in self.doctors:
                specialty = "general_medicine"
            
            available_slots = []
            doctors = self.doctors[specialty]
            
            # Generate slots for next 7 days
            for day in range(1, days_ahead + 1):
                date = datetime.now() + timedelta(days=day)
                
                # Skip weekends for simplicity
                if date.weekday() >= 5:
                    continue
                
                for doctor in doctors:
                    # Generate time slots (9 AM to 5 PM)
                    for hour in range(9, 17, 2):  # Every 2 hours
                        slot_time = date.replace(hour=hour, minute=0, second=0, microsecond=0)
                        slot_id = f"{doctor['id']}_{slot_time.strftime('%Y%m%d_%H%M')}"
                        
                        # Check if slot is already booked
                        if slot_id not in self.appointments:
                            available_slots.append({
                                "slot_id": slot_id,
                                "doctor_id": doctor["id"],
                                "doctor_name": doctor["name"],
                                "specialty": specialty,
                                "time": slot_time.isoformat(),
                                "date": slot_time.strftime("%Y-%m-%d"),
                                "time_str": slot_time.strftime("%I:%M %p")
                            })
            
            # Return first 10 available slots
            return sorted(available_slots, key=lambda x: x["time"])[:10]
            
        except Exception as e:
            print(f"Error getting available slots: {e}")
            return []
    
    def book_appointment(self, patient_id: str, doctor_id: str, slot_time: str, specialty: str) -> Dict:
        """Book an appointment"""
        try:
            # Create appointment ID
            appointment_time = datetime.fromisoformat(slot_time.replace('Z', '+00:00'))
            slot_id = f"{doctor_id}_{appointment_time.strftime('%Y%m%d_%H%M')}"
            
            # Check if slot is available
            if slot_id in self.appointments:
                return {
                    "success": False,
                    "message": "Slot already booked",
                    "slot_id": slot_id
                }
            
            # Find doctor info
            doctor_info = None
            for docs in self.doctors.values():
                for doc in docs:
                    if doc["id"] == doctor_id:
                        doctor_info = doc
                        break
            
            if not doctor_info:
                return {
                    "success": False,
                    "message": "Doctor not found"
                }
            
            # Create appointment
            appointment = {
                "appointment_id": f"apt_{len(self.appointments) + 1:04d}",
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "doctor_name": doctor_info["name"],
                "specialty": specialty,
                "appointment_time": slot_time,
                "status": "confirmed",
                "booked_at": datetime.now().isoformat(),
                "notes": ""
            }
            
            # Save appointment
            self.appointments[slot_id] = appointment
            self._save_appointments()
            
            return {
                "success": True,
                "appointment": appointment,
                "message": f"Appointment booked with {doctor_info['name']} on {appointment_time.strftime('%Y-%m-%d at %I:%M %p')}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Booking failed: {str(e)}"
            }
    
    def get_patient_appointments(self, patient_id: str) -> List[Dict]:
        """Get all appointments for a patient"""
        patient_appointments = []
        
        for appointment in self.appointments.values():
            if appointment.get("patient_id") == patient_id:
                patient_appointments.append(appointment)
        
        return sorted(patient_appointments, key=lambda x: x["appointment_time"])
    
    def cancel_appointment(self, appointment_id: str) -> Dict:
        """Cancel an appointment"""
        try:
            # Find appointment
            for slot_id, appointment in self.appointments.items():
                if appointment.get("appointment_id") == appointment_id:
                    # Remove appointment
                    del self.appointments[slot_id]
                    self._save_appointments()
                    
                    return {
                        "success": True,
                        "message": f"Appointment {appointment_id} cancelled successfully"
                    }
            
            return {
                "success": False,
                "message": "Appointment not found"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Cancellation failed: {str(e)}"
            }
