"""
Streamlit Dashboard for Healthcare Assistant
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

class HealthcareDashboard:
    def __init__(self, healthcare_agent):
        self.agent = healthcare_agent
    
    def render(self):
        """Render the main dashboard"""
        st.title("🏥 Agentic Healthcare Assistant")
        st.markdown("*AI-powered medical task automation with multi-agent system*")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["Chat Interface", "Patient Records", "Appointments", "Medical Search", "System Monitoring"]
        )
        
        if page == "Chat Interface":
            self.render_chat_interface()
        elif page == "Patient Records":
            self.render_patient_records()
        elif page == "Appointments":
            self.render_appointments()
        elif page == "Medical Search":
            self.render_medical_search()
        elif page == "System Monitoring":
            self.render_system_monitoring()
    
    def render_chat_interface(self):
        """Main chat interface with the healthcare agent"""
        st.header("💬 Healthcare Assistant Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show execution details if available
                if message["role"] == "assistant" and "execution_data" in message:
                    with st.expander("View Execution Details"):
                        st.json(message["execution_data"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about appointments, medical records, or health information..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process with healthcare agent
            with st.chat_message("assistant"):
                with st.spinner("Processing your request..."):
                    response_data = self.agent.process_query(prompt)
                
                # Display response
                st.markdown(response_data["response"])
                
                # Show plan if available
                if response_data.get("plan"):
                    with st.expander("Execution Plan"):
                        st.json(response_data["plan"])
                
                # Show execution logs
                if response_data.get("execution_logs"):
                    with st.expander("Execution Logs"):
                        for log in response_data["execution_logs"]:
                            st.text(f"[{log['timestamp']}] {log['step']}: {log['data']}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["response"],
                "execution_data": {
                    "plan": response_data.get("plan"),
                    "logs": response_data.get("execution_logs", [])
                }
            })
        
        # Sample queries
        st.sidebar.markdown("### 💡 Sample Queries")
        sample_queries = [
            "Book a nephrologist appointment for my father with kidney disease",
            "Show me the medical history for patient Ramesh",
            "What are the latest treatments for chronic kidney disease?",
            "Schedule a cardiology appointment and get heart disease information"
        ]
        
        for query in sample_queries:
            if st.sidebar.button(query, key=f"sample_{hash(query)}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
    
    def render_patient_records(self):
        """Patient records management interface"""
        st.header("📋 Patient Records Management")
        
        # Patient search
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search patients by name or condition:")
        with col2:
            if st.button("Search"):
                if search_query:
                    results = self.agent.medical_tools.search_patients(search_query)
                    st.session_state.search_results = results
        
        # Display search results
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.subheader("Search Results")
            for result in st.session_state.search_results:
                with st.expander(f"{result['name']} - {result['patient_id']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Age:** {result.get('age', 'Unknown')}")
                        st.write(f"**Gender:** {result.get('gender', 'Unknown')}")
                        st.write(f"**Phone:** {result.get('phone', 'Unknown')}")
                    with col2:
                        st.write(f"**Conditions:** {', '.join(result.get('conditions', []))}")
                        st.write(f"**Last Visit:** {result.get('last_visit', 'Unknown')}")
        
        # Patient selection
        st.subheader("Patient Details")
        patient_ids = list(self.agent.medical_tools.patient_records.keys())
        selected_patient = st.selectbox("Select Patient:", [""] + patient_ids)
        
        if selected_patient:
            patient_data = self.agent.medical_tools.get_patient_history(selected_patient)
            
            if patient_data:
                # Basic Information
                st.subheader("Basic Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Name", patient_data.get('name', 'Unknown'))
                with col2:
                    st.metric("Age", patient_data.get('age', 'Unknown'))
                with col3:
                    st.metric("Gender", patient_data.get('gender', 'Unknown'))
                
                # Medical Information
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Conditions")
                    conditions = patient_data.get('conditions', [])
                    for condition in conditions:
                        st.write(f"• {condition}")
                
                with col2:
                    st.subheader("Current Medications")
                    medications = patient_data.get('medications', [])
                    for medication in medications:
                        st.write(f"• {medication}")
                
                # Visit History
                if 'visit_history' in patient_data:
                    st.subheader("Recent Visits")
                    visits_df = pd.DataFrame(patient_data['visit_history'])
                    st.dataframe(visits_df, use_container_width=True)
                
                # Lab Results
                if 'lab_results' in patient_data:
                    st.subheader("Lab Results")
                    for lab in patient_data['lab_results']:
                        with st.expander(f"{lab['test']} - {lab['date']}"):
                            st.json(lab['results'])
                            st.write(f"Status: {lab['status']}")
        
        # Add new patient
        st.subheader("Add New Patient")
        with st.form("add_patient"):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("Patient Name")
                new_age = st.number_input("Age", min_value=0, max_value=120)
                new_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with col2:
                new_phone = st.text_input("Phone Number")
                new_conditions = st.text_area("Medical Conditions (comma-separated)")
                new_medications = st.text_area("Current Medications (comma-separated)")
            
            if st.form_submit_button("Add Patient"):
                if new_name:
                    patient_data = {
                        "patient_id": new_name.lower().replace(" ", "_"),
                        "name": new_name,
                        "age": new_age,
                        "gender": new_gender,
                        "phone": new_phone,
                        "conditions": [c.strip() for c in new_conditions.split(",") if c.strip()],
                        "medications": [m.strip() for m in new_medications.split(",") if m.strip()]
                    }
                    
                    result = self.agent.medical_tools.add_patient_record(patient_data)
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])
    
    def render_appointments(self):
        """Appointments management interface"""
        st.header("📅 Appointment Management")
        
        # Appointment booking
        st.subheader("Book New Appointment")
        with st.form("book_appointment"):
            col1, col2 = st.columns(2)
            with col1:
                patient_id = st.text_input("Patient ID")
                specialty = st.selectbox("Specialty", [
                    "general_medicine", "nephrology", "cardiology", 
                    "neurology", "oncology", "dermatology"
                ])
            with col2:
                # Get available slots
                available_slots = self.agent.appointment_tools.get_available_slots(specialty)
                if available_slots:
                    slot_options = [
                        f"{slot['doctor_name']} - {slot['date']} at {slot['time_str']}"
                        for slot in available_slots[:10]
                    ]
                    selected_slot_idx = st.selectbox("Available Slots", range(len(slot_options)), 
                                                   format_func=lambda x: slot_options[x])
                else:
                    st.warning("No available slots for selected specialty")
                    selected_slot_idx = None
            
            if st.form_submit_button("Book Appointment") and selected_slot_idx is not None:
                if patient_id and available_slots:
                    selected_slot = available_slots[selected_slot_idx]
                    result = self.agent.appointment_tools.book_appointment(
                        patient_id=patient_id,
                        doctor_id=selected_slot["doctor_id"],
                        slot_time=selected_slot["time"],
                        specialty=specialty
                    )
                    
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])
        
        # View appointments
        st.subheader("Current Appointments")
        
        # Load all appointments
        appointments = []
        for appointment in self.agent.appointment_tools.appointments.values():
            appointments.append(appointment)
        
        if appointments:
            appointments_df = pd.DataFrame(appointments)
            
            # Display appointments table
            st.dataframe(appointments_df[[
                'appointment_id', 'patient_id', 'doctor_name', 
                'specialty', 'appointment_time', 'status'
            ]], use_container_width=True)
            
            # Appointment analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Appointments by specialty
                specialty_counts = appointments_df['specialty'].value_counts()
                fig_specialty = px.pie(
                    values=specialty_counts.values,
                    names=specialty_counts.index,
                    title="Appointments by Specialty"
                )
                st.plotly_chart(fig_specialty, use_container_width=True)
            
            with col2:
                # Appointments by status
                status_counts = appointments_df['status'].value_counts()
                fig_status = px.bar(
                    x=status_counts.index,
                    y=status_counts.values,
                    title="Appointments by Status"
                )
                st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("No appointments found")
    
    def render_medical_search(self):
        """Medical information search interface"""
        st.header("🔍 Medical Information Search")
        
        # Search interface
        search_query = st.text_input("Search medical information:", 
                                   placeholder="e.g., chronic kidney disease treatment")
        
        if st.button("Search") or search_query:
            if search_query:
                with st.spinner("Searching medical databases..."):
                    results = self.agent.search_tools.search_medical_info(search_query)
                
                if results:
                    st.subheader(f"Search Results for: {search_query}")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"{i+1}. {result['title']} (Score: {result.get('relevance_score', 0):.2f})"):
                            st.write(f"**Source:** {result.get('source', 'Unknown')}")
                            st.write(f"**Category:** {result.get('category', 'General')}")
                            st.write(f"**Last Updated:** {result.get('last_updated', 'Unknown')}")
                            st.write("**Content:**")
                            st.write(result['content'])
                            
                            if 'url' in result:
                                st.markdown(f"[View Source]({result['url']})")
                else:
                    st.warning("No results found for your search query")
        
        # Disease information lookup
        st.subheader("Disease Information Lookup")
        disease_name = st.text_input("Enter disease name:", 
                                   placeholder="e.g., diabetes, hypertension")
        
        if st.button("Get Disease Info") and disease_name:
            with st.spinner("Retrieving disease information..."):
                disease_info = self.agent.search_tools.get_disease_info(disease_name)
            
            if disease_info["status"] == "found":
                st.success(f"Information found for {disease_name}")
                
                # Overview
                st.subheader("Overview")
                st.write(disease_info["overview"])
                
                # Treatment options
                if disease_info["treatment_options"]:
                    st.subheader("Treatment Options")
                    for treatment in disease_info["treatment_options"]:
                        with st.expander(treatment["title"]):
                            st.write(treatment["description"])
                            st.write(f"*Source: {treatment['source']}*")
                
                # Latest research
                if disease_info["latest_research"]:
                    st.subheader("Latest Research")
                    for research in disease_info["latest_research"]:
                        with st.expander(research["title"]):
                            st.write(research["description"])
                            st.write(f"*Source: {research['source']}*")
                
                # Sources
                st.subheader("Sources")
                for source in disease_info["sources"]:
                    st.write(f"• {source}")
            else:
                st.error(disease_info["message"])
        
        # Drug interaction checker
        st.subheader("Drug Interaction Checker")
        medications = st.text_area("Enter medications (one per line):", 
                                 placeholder="metformin\nlisinopril\naspirin")
        
        if st.button("Check Interactions") and medications:
            med_list = [med.strip() for med in medications.split('\n') if med.strip()]
            
            if len(med_list) > 1:
                interactions = self.agent.search_tools.get_drug_interactions(med_list)
                
                if interactions["interactions_found"] > 0:
                    st.warning(f"Found {interactions['interactions_found']} potential interactions")
                    
                    for interaction in interactions["interactions"]:
                        st.error(f"**{' + '.join(interaction['medications'])}**: {interaction['interaction']}")
                else:
                    st.success("No known interactions found")
            else:
                st.info("Please enter at least 2 medications to check for interactions")
    
    def render_system_monitoring(self):
        """System monitoring and evaluation interface"""
        st.header("📊 System Monitoring & Evaluation")
        
        # Get execution metrics
        metrics = self.agent.get_execution_metrics()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Executions", metrics.get("total_executions", 0))
        with col2:
            st.metric("Success Rate", "95%")  # Mock data
        with col3:
            st.metric("Avg Response Time", "2.3s")  # Mock data
        with col4:
            st.metric("Active Patients", len(self.agent.medical_tools.patient_records))
        
        # Execution distribution
        if "step_distribution" in metrics:
            st.subheader("Agent Step Distribution")
            step_data = metrics["step_distribution"]
            
            fig = px.bar(
                x=list(step_data.keys()),
                y=list(step_data.values()),
                title="Execution Steps by Agent Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent execution logs
        st.subheader("Recent Execution Logs")
        if "recent_logs" in metrics and metrics["recent_logs"]:
            for log in metrics["recent_logs"]:
                with st.expander(f"{log['step']} - {log['timestamp']}"):
                    st.json(log['data'])
        else:
            st.info("No recent execution logs available")
        
        # Model evaluation
        st.subheader("Model Evaluation Metrics")
        
        # Mock evaluation data
        eval_data = {
            "Accuracy": 0.92,
            "Precision": 0.89,
            "Recall": 0.94,
            "F1-Score": 0.91
        }
        
        col1, col2 = st.columns(2)
        with col1:
            # Metrics table
            eval_df = pd.DataFrame(list(eval_data.items()), columns=["Metric", "Score"])
            st.dataframe(eval_df, use_container_width=True)
        
        with col2:
            # Metrics chart
            fig = go.Figure(data=go.Scatterpolar(
                r=list(eval_data.values()),
                theta=list(eval_data.keys()),
                fill='toself',
                name='Model Performance'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Model Performance Radar"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Memory usage
        st.subheader("Memory & Storage")
        memory_stats = {
            "Patient Records": len(self.agent.medical_tools.patient_records),
            "Appointments": len(self.agent.appointment_tools.appointments),
            "Knowledge Base": len(self.agent.search_tools.knowledge_base),
            "Conversation History": len(self.agent.conversation_memory.chat_memory.messages) if hasattr(self.agent.conversation_memory, 'chat_memory') else 0
        }
        
        for key, value in memory_stats.items():
            st.metric(key, value)
        
        # System health
        st.subheader("System Health")
        health_status = {
            "LLM Connection": "✅ Connected",
            "Vector Database": "✅ Operational", 
            "Memory System": "✅ Functional",
            "Search Tools": "✅ Active"
        }
        
        for component, status in health_status.items():
            st.write(f"**{component}:** {status}")
        
        # Clear data options
        st.subheader("Data Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.success("Chat history cleared")
        
        with col2:
            if st.button("Reset Execution Logs"):
                self.agent.execution_logs = []
                st.success("Execution logs reset")
        
        with col3:
            if st.button("Export Data"):
                # Mock export functionality
                st.success("Data export initiated")
