"""
Enhanced Medical Records Agent with LLM Summarization and Structured Processing
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class EnhancedRecordsAgent:
    def __init__(self, llm, medical_tools):
        self.llm = llm
        self.tools = medical_tools
        
        self.summarization_prompt_template = """
You are a medical records specialist. Analyze and summarize patient medical information comprehensively.

Patient Data: {patient_data}
Query Context: {query_context}

Provide a structured medical summary including:

1. PATIENT OVERVIEW
   - Demographics and basic information
   - Primary medical concerns

2. MEDICAL HISTORY
   - Chronic conditions and diagnoses
   - Significant past medical events
   - Family medical history (if available)

3. CURRENT MEDICATIONS
   - Active prescriptions
   - Dosages and frequencies
   - Potential interactions or concerns

4. RECENT MEDICAL ACTIVITY
   - Recent visits and consultations
   - Laboratory results and trends
   - Diagnostic procedures

5. CLINICAL ALERTS
   - Allergies and adverse reactions
   - Risk factors and warnings
   - Follow-up requirements

6. CARE RECOMMENDATIONS
   - Suggested monitoring
   - Lifestyle modifications
   - Specialist referrals needed

Format as a professional medical summary suitable for healthcare providers.
Include relevant medical terminology while remaining clear and actionable.
"""
    
    def execute_with_llm_summary(self, plan: Dict, patient_context: Optional[Dict] = None) -> Dict:
        """Execute medical records retrieval with comprehensive LLM summarization"""
        try:
            # Extract patient identification from plan
            patient_id = plan.get("patient_id", "unknown")
            
            # Retrieve comprehensive patient data
            patient_data = self.tools.get_patient_history(patient_id)
            
            if not patient_data:
                return {
                    "status": "not_found",
                    "message": f"No medical records found for patient: {patient_id}",
                    "patient_id": patient_id,
                    "search_attempted": True
                }
            
            # Generate comprehensive LLM summary
            medical_summary = self._generate_comprehensive_summary(
                patient_data, 
                plan, 
                patient_context
            )
            
            # Extract key clinical insights
            clinical_insights = self._extract_clinical_insights(patient_data, medical_summary)
            
            # Generate care recommendations
            care_recommendations = self._generate_care_recommendations(
                patient_data, 
                medical_summary
            )
            
            return {
                "status": "success",
                "patient_id": patient_id,
                "patient_data": patient_data,
                "medical_summary": medical_summary,
                "clinical_insights": clinical_insights,
                "care_recommendations": care_recommendations,
                "summary_generated_by": "enhanced_llm_agent",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Medical records processing failed: {str(e)}",
                "patient_id": plan.get("patient_id", "unknown")
            }
    
    def _generate_comprehensive_summary(self, patient_data: Dict, 
                                      plan: Dict, 
                                      patient_context: Optional[Dict]) -> str:
        """Generate comprehensive medical summary using LLM"""
        try:
            # Prepare context for summarization
            query_context = {
                "original_query": plan.get("query_analysis", {}).get("original_query", ""),
                "medical_domains": plan.get("query_analysis", {}).get("medical_domains", []),
                "patient_context": patient_context
            }
            
            # Format prompt
            formatted_prompt = self.summarization_prompt_template.format(
                patient_data=json.dumps(patient_data, indent=2),
                query_context=json.dumps(query_context, indent=2)
            )
            
            # Generate summary
            response = self.llm.invoke(formatted_prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            
            # Enhance summary with structured data
            enhanced_summary = self._enhance_summary_with_structure(summary, patient_data)
            
            return enhanced_summary
            
        except Exception as e:
            # Fallback to structured summary
            return self._create_structured_fallback_summary(patient_data)
    
    def _enhance_summary_with_structure(self, llm_summary: str, patient_data: Dict) -> str:
        """Enhance LLM summary with structured data elements"""
        try:
            # Add structured data sections
            structured_additions = []
            
            # Add vital statistics if available
            if 'lab_results' in patient_data:
                structured_additions.append("\n📊 RECENT LAB RESULTS:")
                for lab in patient_data['lab_results'][-2:]:  # Last 2 results
                    structured_additions.append(f"• {lab.get('test', 'Unknown Test')} ({lab.get('date', 'Unknown Date')}): {lab.get('status', 'Unknown Status')}")
            
            # Add visit timeline
            if 'visit_history' in patient_data:
                structured_additions.append("\n📅 RECENT VISITS:")
                for visit in patient_data['visit_history'][-3:]:  # Last 3 visits
                    structured_additions.append(f"• {visit.get('date', 'Unknown Date')}: {visit.get('type', 'Unknown Type')} - {visit.get('diagnosis', 'No diagnosis recorded')}")
            
            # Add medication details
            medications = patient_data.get('medications', [])
            if medications:
                structured_additions.append("\n💊 CURRENT MEDICATIONS:")
                for med in medications:
                    if med.strip():  # Only non-empty medications
                        structured_additions.append(f"• {med}")
            
            # Combine LLM summary with structured additions
            enhanced_summary = llm_summary
            if structured_additions:
                enhanced_summary += "\n\n" + "\n".join(structured_additions)
            
            return enhanced_summary
            
        except Exception as e:
            return llm_summary  # Return original if enhancement fails
    
    def _extract_clinical_insights(self, patient_data: Dict, medical_summary: str) -> Dict:
        """Extract key clinical insights from patient data and summary"""
        try:
            insights_prompt = f"""
            Extract key clinical insights from this medical information:
            
            Patient Data: {json.dumps(patient_data, indent=2)}
            Medical Summary: {medical_summary}
            
            Identify and return JSON with:
            {{
                "risk_factors": ["list of identified risk factors"],
                "chronic_conditions": ["list of chronic conditions"],
                "medication_concerns": ["potential medication issues"],
                "follow_up_needs": ["required follow-up actions"],
                "care_gaps": ["identified gaps in care"],
                "priority_alerts": ["high priority medical alerts"]
            }}
            """
            
            response = self.llm.invoke(insights_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    insights = json.loads(json_match.group())
                    return insights
                except json.JSONDecodeError:
                    pass
            
            # Fallback to rule-based insights
            return self._generate_rule_based_insights(patient_data)
            
        except Exception as e:
            return self._generate_rule_based_insights(patient_data)
    
    def _generate_rule_based_insights(self, patient_data: Dict) -> Dict:
        """Generate clinical insights using rule-based approach"""
        insights = {
            "risk_factors": [],
            "chronic_conditions": [],
            "medication_concerns": [],
            "follow_up_needs": [],
            "care_gaps": [],
            "priority_alerts": []
        }
        
        # Analyze conditions
        conditions = patient_data.get('conditions', [])
        for condition in conditions:
            condition_lower = condition.lower()
            
            if any(chronic in condition_lower for chronic in ['diabetes', 'hypertension', 'kidney', 'heart']):
                insights["chronic_conditions"].append(condition)
                insights["follow_up_needs"].append(f"Regular monitoring for {condition}")
            
            if 'diabetes' in condition_lower:
                insights["risk_factors"].append("Cardiovascular disease risk")
                insights["care_gaps"].append("HbA1c monitoring")
            
            if 'hypertension' in condition_lower:
                insights["risk_factors"].append("Stroke and heart disease risk")
                insights["care_gaps"].append("Blood pressure monitoring")
        
        # Analyze age-related risks
        age = patient_data.get('age')
        if age and isinstance(age, int):
            if age > 65:
                insights["risk_factors"].append("Age-related health risks")
                insights["follow_up_needs"].append("Annual comprehensive geriatric assessment")
        
        # Check for medication interactions
        medications = patient_data.get('medications', [])
        if len(medications) > 3:
            insights["medication_concerns"].append("Polypharmacy - review for interactions")
        
        return insights
    
    def _generate_care_recommendations(self, patient_data: Dict, medical_summary: str) -> Dict:
        """Generate personalized care recommendations"""
        try:
            recommendations_prompt = f"""
            Based on this patient's medical information, provide personalized care recommendations:
            
            Patient Data: {json.dumps(patient_data, indent=2)}
            Medical Summary: {medical_summary}
            
            Generate recommendations in JSON format:
            {{
                "immediate_actions": ["urgent actions needed"],
                "routine_monitoring": ["regular monitoring requirements"],
                "lifestyle_modifications": ["recommended lifestyle changes"],
                "specialist_referrals": ["specialist consultations needed"],
                "preventive_care": ["preventive measures to implement"],
                "patient_education": ["key education topics for patient"]
            }}
            
            Base recommendations on evidence-based guidelines and patient-specific factors.
            """
            
            response = self.llm.invoke(recommendations_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    recommendations = json.loads(json_match.group())
                    return recommendations
                except json.JSONDecodeError:
                    pass
            
            # Fallback to rule-based recommendations
            return self._generate_rule_based_recommendations(patient_data)
            
        except Exception as e:
            return self._generate_rule_based_recommendations(patient_data)
    
    def _generate_rule_based_recommendations(self, patient_data: Dict) -> Dict:
        """Generate care recommendations using rule-based approach"""
        recommendations = {
            "immediate_actions": [],
            "routine_monitoring": [],
            "lifestyle_modifications": [],
            "specialist_referrals": [],
            "preventive_care": [],
            "patient_education": []
        }
        
        conditions = patient_data.get('conditions', [])
        age = patient_data.get('age', 0)
        
        for condition in conditions:
            condition_lower = condition.lower()
            
            if 'diabetes' in condition_lower:
                recommendations["routine_monitoring"].append("HbA1c every 3-6 months")
                recommendations["lifestyle_modifications"].append("Dietary counseling and exercise program")
                recommendations["patient_education"].append("Diabetes self-management education")
            
            if 'hypertension' in condition_lower:
                recommendations["routine_monitoring"].append("Blood pressure monitoring")
                recommendations["lifestyle_modifications"].append("DASH diet and sodium restriction")
                recommendations["patient_education"].append("Blood pressure self-monitoring")
            
            if 'kidney' in condition_lower:
                recommendations["specialist_referrals"].append("Nephrology consultation")
                recommendations["routine_monitoring"].append("Kidney function tests")
                recommendations["patient_education"].append("Chronic kidney disease management")
        
        # Age-based recommendations
        if isinstance(age, int) and age > 65:
            recommendations["preventive_care"].append("Annual influenza vaccination")
            recommendations["preventive_care"].append("Pneumococcal vaccination")
            recommendations["routine_monitoring"].append("Fall risk assessment")
        
        return recommendations
    
    def _create_structured_fallback_summary(self, patient_data: Dict) -> str:
        """Create structured fallback summary when LLM fails"""
        summary_parts = []
        
        # Patient Overview
        summary_parts.append("📋 PATIENT MEDICAL SUMMARY")
        summary_parts.append("=" * 40)
        
        # Basic Information
        summary_parts.append(f"\n👤 PATIENT: {patient_data.get('name', 'Unknown')}")
        summary_parts.append(f"Age: {patient_data.get('age', 'Unknown')}")
        summary_parts.append(f"Gender: {patient_data.get('gender', 'Unknown')}")
        
        # Medical Conditions
        conditions = patient_data.get('conditions', [])
        if conditions:
            summary_parts.append(f"\n🏥 MEDICAL CONDITIONS:")
            for condition in conditions:
                if condition.strip():
                    summary_parts.append(f"• {condition}")
        
        # Current Medications
        medications = patient_data.get('medications', [])
        if medications:
            summary_parts.append(f"\n💊 CURRENT MEDICATIONS:")
            for medication in medications:
                if medication.strip():
                    summary_parts.append(f"• {medication}")
        
        # Recent Activity
        if 'visit_history' in patient_data:
            summary_parts.append(f"\n📅 RECENT VISITS:")
            for visit in patient_data['visit_history'][-2:]:
                summary_parts.append(f"• {visit.get('date', 'Unknown')}: {visit.get('diagnosis', 'Routine visit')}")
        
        summary_parts.append(f"\n⚠️ This summary was generated automatically. Please verify all information with the patient and medical records.")
        
        return "\n".join(summary_parts)
