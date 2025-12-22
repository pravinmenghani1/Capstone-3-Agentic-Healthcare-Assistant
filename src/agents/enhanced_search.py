"""
Enhanced Search Agent with Advanced RAG Pipeline and Medical Knowledge Integration
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class EnhancedSearchAgent:
    def __init__(self, llm, search_tools):
        self.llm = llm
        self.tools = search_tools
        
        self.rag_prompt_template = """
You are a medical information specialist with access to comprehensive medical databases. 
Provide evidence-based medical information using RAG (Retrieval-Augmented Generation).

Query Context: {query_context}
Patient Context: {patient_context}
Retrieved Medical Sources: {retrieved_sources}

Your task:
1. Analyze the medical query in context of patient information
2. Synthesize information from retrieved sources
3. Provide comprehensive, evidence-based medical information
4. Include current treatment guidelines and recommendations
5. Consider patient-specific factors and contraindications

Structure your response as:

🔍 MEDICAL INFORMATION SUMMARY
- Primary condition/topic overview
- Current understanding and classification

📋 EVIDENCE-BASED TREATMENT OPTIONS
- First-line treatments with evidence levels
- Alternative therapies and considerations
- Patient-specific modifications needed

⚠️ IMPORTANT CONSIDERATIONS
- Contraindications and warnings
- Drug interactions (if applicable)
- Monitoring requirements

📚 CURRENT GUIDELINES & RESEARCH
- Latest clinical guidelines (year and source)
- Recent research findings
- Emerging therapies

🏥 CLINICAL RECOMMENDATIONS
- Recommended next steps
- Specialist referrals if needed
- Patient education priorities

Always include appropriate medical disclaimers and emphasize the need for professional medical consultation.
"""
    
    def execute_rag_search(self, plan: Dict, patient_context: Optional[Dict] = None) -> Dict:
        """Execute advanced RAG-based medical information search"""
        try:
            # Extract search requirements from plan
            search_requirements = self._analyze_search_requirements(plan, patient_context)
            
            # Perform multi-source medical information retrieval
            retrieved_sources = self._perform_comprehensive_search(search_requirements)
            
            if not retrieved_sources:
                return {
                    "status": "no_results",
                    "message": f"No medical information found for: {search_requirements['primary_query']}",
                    "search_query": search_requirements['primary_query']
                }
            
            # Generate comprehensive RAG response
            rag_response = self._generate_rag_response(
                search_requirements,
                retrieved_sources,
                patient_context
            )
            
            # Extract key medical insights
            medical_insights = self._extract_medical_insights(
                retrieved_sources,
                rag_response
            )
            
            # Generate treatment recommendations
            treatment_recommendations = self._generate_treatment_recommendations(
                search_requirements,
                retrieved_sources,
                patient_context
            )
            
            return {
                "status": "success",
                "search_query": search_requirements['primary_query'],
                "retrieved_sources": retrieved_sources,
                "rag_response": rag_response,
                "medical_insights": medical_insights,
                "treatment_recommendations": treatment_recommendations,
                "evidence_level": self._assess_evidence_level(retrieved_sources),
                "patient_specific": patient_context is not None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Medical search failed: {str(e)}",
                "search_attempted": True
            }
    
    def _analyze_search_requirements(self, plan: Dict, patient_context: Optional[Dict]) -> Dict:
        """Analyze search requirements from plan and context"""
        requirements = {
            "primary_query": "",
            "medical_domains": [],
            "search_type": "general",
            "patient_specific_factors": [],
            "urgency_level": "routine"
        }
        
        # Extract search query from plan tasks
        tasks = plan.get("tasks", [])
        query_parts = []
        
        for task in tasks:
            if any(keyword in task.lower() for keyword in ["search", "information", "treatment", "latest"]):
                query_parts.append(task)
        
        # Combine query parts or use plan intent
        if query_parts:
            requirements["primary_query"] = " ".join(query_parts)
        else:
            requirements["primary_query"] = plan.get("intent", "general medical information")
        
        # Extract medical domains from plan
        requirements["medical_domains"] = plan.get("query_analysis", {}).get("medical_domains", ["general_medicine"])
        
        # Determine search type
        query_lower = requirements["primary_query"].lower()
        if any(word in query_lower for word in ["treatment", "therapy", "medication"]):
            requirements["search_type"] = "treatment_focused"
        elif any(word in query_lower for word in ["diagnosis", "symptoms", "condition"]):
            requirements["search_type"] = "diagnostic_focused"
        elif any(word in query_lower for word in ["latest", "recent", "new", "research"]):
            requirements["search_type"] = "research_focused"
        
        # Extract patient-specific factors
        if patient_context:
            patient_data = patient_context.get("data", {})
            
            # Add age considerations
            age = patient_data.get("age")
            if age:
                if isinstance(age, int):
                    if age > 65:
                        requirements["patient_specific_factors"].append("geriatric_considerations")
                    elif age < 18:
                        requirements["patient_specific_factors"].append("pediatric_considerations")
            
            # Add condition-specific factors
            conditions = patient_data.get("conditions", [])
            for condition in conditions:
                condition_lower = condition.lower()
                if "kidney" in condition_lower:
                    requirements["patient_specific_factors"].append("renal_impairment")
                elif "liver" in condition_lower:
                    requirements["patient_specific_factors"].append("hepatic_impairment")
                elif "diabetes" in condition_lower:
                    requirements["patient_specific_factors"].append("diabetes_considerations")
        
        return requirements
    
    def _perform_comprehensive_search(self, requirements: Dict) -> List[Dict]:
        """Perform comprehensive multi-source medical search"""
        try:
            # Primary search using existing tools
            primary_results = self.tools.search_medical_info(requirements["primary_query"])
            
            # Enhanced search for specific domains
            enhanced_results = []
            for domain in requirements["medical_domains"]:
                domain_query = f"{requirements['primary_query']} {domain}"
                domain_results = self.tools.search_medical_info(domain_query)
                enhanced_results.extend(domain_results)
            
            # Search for patient-specific information
            patient_specific_results = []
            for factor in requirements["patient_specific_factors"]:
                factor_query = f"{requirements['primary_query']} {factor}"
                factor_results = self.tools.search_medical_info(factor_query)
                patient_specific_results.extend(factor_results)
            
            # Combine and deduplicate results
            all_results = primary_results + enhanced_results + patient_specific_results
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_results = []
            for result in all_results:
                title = result.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(result)
            
            # Sort by relevance score
            unique_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            return unique_results[:10]  # Top 10 most relevant results
            
        except Exception as e:
            # Fallback to basic search
            return self.tools.search_medical_info(requirements["primary_query"])
    
    def _generate_rag_response(self, requirements: Dict, 
                             retrieved_sources: List[Dict], 
                             patient_context: Optional[Dict]) -> str:
        """Generate comprehensive RAG response using retrieved sources"""
        try:
            # Prepare context for RAG prompt
            query_context = {
                "primary_query": requirements["primary_query"],
                "medical_domains": requirements["medical_domains"],
                "search_type": requirements["search_type"],
                "patient_factors": requirements["patient_specific_factors"]
            }
            
            # Format retrieved sources
            sources_text = []
            for i, source in enumerate(retrieved_sources[:5], 1):  # Top 5 sources
                sources_text.append(f"""
Source {i}: {source.get('title', 'Unknown Title')}
Publisher: {source.get('source', 'Unknown Source')}
Content: {source.get('content', 'No content available')}
Relevance Score: {source.get('relevance_score', 0)}
""")
            
            # Format patient context
            patient_context_str = json.dumps(patient_context, indent=2) if patient_context else "No patient-specific context"
            
            # Generate RAG response
            formatted_prompt = self.rag_prompt_template.format(
                query_context=json.dumps(query_context, indent=2),
                patient_context=patient_context_str,
                retrieved_sources="\n".join(sources_text)
            )
            
            response = self.llm.invoke(formatted_prompt)
            rag_response = response.content if hasattr(response, 'content') else str(response)
            
            # Enhance response with source citations
            enhanced_response = self._add_source_citations(rag_response, retrieved_sources)
            
            return enhanced_response
            
        except Exception as e:
            # Fallback to structured summary
            return self._create_structured_fallback_response(requirements, retrieved_sources)
    
    def _add_source_citations(self, rag_response: str, sources: List[Dict]) -> str:
        """Add source citations to RAG response"""
        try:
            citation_section = "\n\n📚 SOURCES AND REFERENCES:\n"
            
            for i, source in enumerate(sources[:5], 1):
                citation = f"[{i}] {source.get('title', 'Unknown Title')} - {source.get('source', 'Unknown Source')}"
                if 'last_updated' in source:
                    citation += f" ({source['last_updated']})"
                citation_section += f"{citation}\n"
            
            return rag_response + citation_section
            
        except Exception as e:
            return rag_response
    
    def _extract_medical_insights(self, sources: List[Dict], rag_response: str) -> Dict:
        """Extract key medical insights from sources and response"""
        try:
            insights_prompt = f"""
            Extract key medical insights from the following information:
            
            RAG Response: {rag_response}
            Source Count: {len(sources)}
            
            Identify and return JSON with:
            {{
                "key_findings": ["most important medical findings"],
                "treatment_efficacy": ["treatment effectiveness information"],
                "safety_profile": ["safety and side effect information"],
                "contraindications": ["important contraindications"],
                "drug_interactions": ["significant drug interactions"],
                "monitoring_requirements": ["required monitoring parameters"],
                "evidence_strength": "strong/moderate/limited/insufficient"
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
            return self._generate_rule_based_medical_insights(sources, rag_response)
            
        except Exception as e:
            return self._generate_rule_based_medical_insights(sources, rag_response)
    
    def _generate_rule_based_medical_insights(self, sources: List[Dict], rag_response: str) -> Dict:
        """Generate medical insights using rule-based approach"""
        insights = {
            "key_findings": [],
            "treatment_efficacy": [],
            "safety_profile": [],
            "contraindications": [],
            "drug_interactions": [],
            "monitoring_requirements": [],
            "evidence_strength": "moderate"
        }
        
        # Analyze source content for insights
        all_content = " ".join([source.get("content", "") for source in sources])
        response_content = rag_response.lower()
        
        # Extract key findings
        if "effective" in response_content or "efficacy" in response_content:
            insights["treatment_efficacy"].append("Treatment shows documented efficacy")
        
        if "side effect" in response_content or "adverse" in response_content:
            insights["safety_profile"].append("Monitor for potential side effects")
        
        if "contraindicated" in response_content or "avoid" in response_content:
            insights["contraindications"].append("Important contraindications identified")
        
        if "monitor" in response_content or "follow-up" in response_content:
            insights["monitoring_requirements"].append("Regular monitoring recommended")
        
        # Assess evidence strength based on source count and quality
        if len(sources) >= 5:
            insights["evidence_strength"] = "strong"
        elif len(sources) >= 3:
            insights["evidence_strength"] = "moderate"
        else:
            insights["evidence_strength"] = "limited"
        
        return insights
    
    def _generate_treatment_recommendations(self, requirements: Dict, 
                                         sources: List[Dict], 
                                         patient_context: Optional[Dict]) -> Dict:
        """Generate personalized treatment recommendations"""
        try:
            recommendations_prompt = f"""
            Based on the medical information and patient context, provide treatment recommendations:
            
            Search Requirements: {json.dumps(requirements, indent=2)}
            Patient Context: {json.dumps(patient_context, indent=2) if patient_context else "No patient context"}
            Available Sources: {len(sources)} medical sources
            
            Generate recommendations in JSON format:
            {{
                "first_line_treatments": ["primary treatment options"],
                "alternative_therapies": ["alternative treatment approaches"],
                "lifestyle_modifications": ["recommended lifestyle changes"],
                "monitoring_plan": ["monitoring requirements and schedule"],
                "patient_education": ["key education points"],
                "follow_up_timeline": ["recommended follow-up schedule"],
                "specialist_consultation": ["when to refer to specialists"],
                "contraindications_check": ["important contraindications to verify"]
            }}
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
            return self._generate_rule_based_treatment_recommendations(requirements, patient_context)
            
        except Exception as e:
            return self._generate_rule_based_treatment_recommendations(requirements, patient_context)
    
    def _generate_rule_based_treatment_recommendations(self, requirements: Dict, 
                                                     patient_context: Optional[Dict]) -> Dict:
        """Generate treatment recommendations using rule-based approach"""
        recommendations = {
            "first_line_treatments": ["Consult healthcare provider for personalized treatment plan"],
            "alternative_therapies": ["Discuss alternative options with healthcare team"],
            "lifestyle_modifications": ["Maintain healthy diet and regular exercise"],
            "monitoring_plan": ["Regular follow-up with healthcare provider"],
            "patient_education": ["Understand condition and treatment options"],
            "follow_up_timeline": ["Schedule follow-up as recommended by provider"],
            "specialist_consultation": ["Consider specialist referral if indicated"],
            "contraindications_check": ["Review all medications and allergies with provider"]
        }
        
        # Add domain-specific recommendations
        for domain in requirements.get("medical_domains", []):
            if domain == "nephrology":
                recommendations["monitoring_plan"].append("Regular kidney function monitoring")
                recommendations["lifestyle_modifications"].append("Protein and sodium restriction as advised")
            elif domain == "cardiology":
                recommendations["monitoring_plan"].append("Regular blood pressure and heart rate monitoring")
                recommendations["lifestyle_modifications"].append("Heart-healthy diet and exercise program")
        
        # Add patient-specific considerations
        if patient_context:
            patient_data = patient_context.get("data", {})
            age = patient_data.get("age")
            
            if age and isinstance(age, int) and age > 65:
                recommendations["monitoring_plan"].append("Enhanced monitoring for age-related considerations")
                recommendations["contraindications_check"].append("Review age-appropriate dosing")
        
        return recommendations
    
    def _assess_evidence_level(self, sources: List[Dict]) -> str:
        """Assess overall evidence level of retrieved sources"""
        if not sources:
            return "insufficient"
        
        # Simple assessment based on source count and types
        source_count = len(sources)
        
        # Check for high-quality sources
        high_quality_sources = 0
        for source in sources:
            source_name = source.get("source", "").lower()
            if any(quality_indicator in source_name for quality_indicator in 
                   ["journal", "guidelines", "association", "foundation"]):
                high_quality_sources += 1
        
        if source_count >= 5 and high_quality_sources >= 2:
            return "strong"
        elif source_count >= 3 and high_quality_sources >= 1:
            return "moderate"
        elif source_count >= 2:
            return "limited"
        else:
            return "insufficient"
    
    def _create_structured_fallback_response(self, requirements: Dict, sources: List[Dict]) -> str:
        """Create structured fallback response when LLM fails"""
        response_parts = []
        
        response_parts.append("🔍 MEDICAL INFORMATION SEARCH RESULTS")
        response_parts.append("=" * 50)
        
        response_parts.append(f"\n📋 SEARCH QUERY: {requirements['primary_query']}")
        response_parts.append(f"Medical Domains: {', '.join(requirements['medical_domains'])}")
        response_parts.append(f"Sources Found: {len(sources)}")
        
        if sources:
            response_parts.append(f"\n📚 TOP MEDICAL SOURCES:")
            for i, source in enumerate(sources[:3], 1):
                response_parts.append(f"{i}. {source.get('title', 'Unknown Title')}")
                response_parts.append(f"   Source: {source.get('source', 'Unknown Source')}")
                response_parts.append(f"   Summary: {source.get('content', 'No summary available')[:200]}...")
        
        response_parts.append(f"\n⚠️ MEDICAL DISCLAIMER:")
        response_parts.append("This information is for educational purposes only.")
        response_parts.append("Always consult with qualified healthcare professionals for medical advice.")
        
        return "\n".join(response_parts)
