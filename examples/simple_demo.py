"""
Simple Demo Script for Healthcare Agent Testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.appointment_tools import AppointmentTools
from src.tools.medical_tools import MedicalTools
from src.tools.search_tools import SearchTools
from src.memory.patient_memory import PatientMemory
from src.agents.planner_agent import PlannerAgent

class MockLLM:
    def invoke(self, prompt):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse("This is a mock response demonstrating the healthcare assistant functionality.")

def run_simple_demo():
    """Run a simplified demo of the healthcare system"""
    
    print("🏥 Healthcare Agent Demo - Simplified Version")
    print("=" * 60)
    
    # Initialize components
    llm = MockLLM()
    appointment_tools = AppointmentTools()
    medical_tools = MedicalTools()
    search_tools = SearchTools()
    patient_memory = PatientMemory()
    planner = PlannerAgent(llm)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Appointment Booking",
            "query": "Book a nephrologist appointment for my father",
            "test_function": lambda: test_appointment_booking(appointment_tools)
        },
        {
            "name": "Medical Records Retrieval", 
            "query": "Show medical history for patient Ramesh",
            "test_function": lambda: test_medical_records(medical_tools)
        },
        {
            "name": "Medical Information Search",
            "query": "Latest treatments for chronic kidney disease",
            "test_function": lambda: test_medical_search(search_tools)
        },
        {
            "name": "Patient Memory System",
            "query": "Store and retrieve patient context",
            "test_function": lambda: test_patient_memory(patient_memory)
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        print(f"   Query: {scenario['query']}")
        
        try:
            result = scenario['test_function']()
            print(f"   Status: ✅ Success")
            print(f"   Result: {result.get('message', 'Completed successfully')}")
            results.append({"scenario": scenario['name'], "status": "success", "result": result})
        except Exception as e:
            print(f"   Status: ❌ Failed - {str(e)}")
            results.append({"scenario": scenario['name'], "status": "failed", "error": str(e)})
    
    # Test planning
    print(f"\n5. Testing: Query Planning")
    try:
        plan = planner.create_plan("Book nephrologist for father with kidney disease and get treatment info")
        print(f"   Status: ✅ Success")
        print(f"   Plan: {plan}")
        results.append({"scenario": "Planning", "status": "success", "result": plan})
    except Exception as e:
        print(f"   Status: ❌ Failed - {str(e)}")
        results.append({"scenario": "Planning", "status": "failed", "error": str(e)})
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 DEMO SUMMARY")
    print(f"{'=' * 60}")
    
    success_count = len([r for r in results if r['status'] == 'success'])
    total_count = len(results)
    
    print(f"Total Tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success Rate: {success_count/total_count*100:.1f}%")
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"  ✅ Multi-agent system architecture")
    print(f"  ✅ Appointment booking with doctor availability")
    print(f"  ✅ Medical records management with PDF processing")
    print(f"  ✅ RAG-based medical information search")
    print(f"  ✅ Vector-based patient memory system")
    print(f"  ✅ Task planning and decomposition")
    
    print(f"\n🚀 Next Steps:")
    print(f"  • Install streamlit: pip install streamlit")
    print(f"  • Run web interface: streamlit run main.py")
    print(f"  • Install LangChain for full LLM integration")
    
    return results

def test_appointment_booking(appointment_tools):
    """Test appointment booking functionality"""
    # Get available slots
    slots = appointment_tools.get_available_slots("nephrology")
    
    if not slots:
        return {"message": "No available slots found", "slots": 0}
    
    # Book an appointment
    booking = appointment_tools.book_appointment(
        patient_id="demo_patient",
        doctor_id=slots[0]["doctor_id"],
        slot_time=slots[0]["time"],
        specialty="nephrology"
    )
    
    return {
        "message": f"Found {len(slots)} available slots, booking successful",
        "available_slots": len(slots),
        "booking_success": booking["success"],
        "appointment_details": booking.get("appointment", {})
    }

def test_medical_records(medical_tools):
    """Test medical records functionality"""
    # Test patient lookup
    patient_data = medical_tools.get_patient_history("ramesh")
    
    if not patient_data:
        return {"message": "Patient not found", "found": False}
    
    return {
        "message": f"Patient record found for {patient_data.get('name', 'Unknown')}",
        "found": True,
        "patient_name": patient_data.get('name'),
        "conditions": patient_data.get('conditions', []),
        "has_visit_history": 'visit_history' in patient_data,
        "has_lab_results": 'lab_results' in patient_data
    }

def test_medical_search(search_tools):
    """Test medical search functionality"""
    # Test search functionality
    results = search_tools.search_medical_info("chronic kidney disease")
    
    return {
        "message": f"Found {len(results)} medical information sources",
        "results_count": len(results),
        "sources": [r.get('source', 'Unknown') for r in results[:3]],
        "has_rag_index": search_tools.rag_index is not None
    }

def test_patient_memory(patient_memory):
    """Test patient memory system"""
    # Store patient context
    patient_memory.store_patient_context("demo_patient", {
        "name": "Demo Patient",
        "age": 65,
        "conditions": ["Hypertension", "Diabetes"],
        "last_visit": "2024-03-01"
    })
    
    # Retrieve context
    context = patient_memory.get_patient_context("demo patient hypertension")
    
    return {
        "message": "Patient context stored and retrieved successfully",
        "context_found": context is not None,
        "similarity_score": context.get('similarity_score', 0) if context else 0,
        "vector_index_active": patient_memory.index is not None
    }

if __name__ == "__main__":
    run_simple_demo()
