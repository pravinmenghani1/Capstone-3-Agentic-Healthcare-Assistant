"""
Enhanced Planner Agent with Proper Goal Decomposition and Prompt Engineering
"""

from typing import Dict, List, Any, Optional
import json
import re

class EnhancedPlannerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.planning_templates = {
            "multi_task": """
You are an expert healthcare planning agent. Analyze the complex healthcare query and create a detailed execution plan.

User Query: {query}
Patient Context: {context}

Your task is to:
1. Identify the patient (father, mother, self, named patient)
2. Determine the primary healthcare intent
3. Break down the query into sequential, actionable sub-tasks
4. Identify required tools and APIs for each task
5. Set appropriate priority and execution order

Return a JSON plan with this exact structure:
{{
    "patient_id": "extracted patient identifier",
    "intent": "primary healthcare objective",
    "tasks": [
        "retrieve_patient_context",
        "specific_task_1",
        "specific_task_2"
    ],
    "priority": "high/medium/low",
    "estimated_steps": number,
    "required_tools": ["memory", "appointment", "records", "search"],
    "execution_order": "sequential/parallel",
    "context_dependencies": ["task dependencies"]
}}

Example for "Book nephrologist for my father with kidney disease, also get treatment info":
{{
    "patient_id": "father",
    "intent": "book_specialist_appointment_and_research_treatment",
    "tasks": [
        "retrieve_patient_context",
        "book_nephrologist_appointment",
        "search_kidney_disease_treatments",
        "synthesize_comprehensive_response"
    ],
    "priority": "high",
    "estimated_steps": 4,
    "required_tools": ["memory", "appointment", "search"],
    "execution_order": "sequential",
    "context_dependencies": ["patient_context_for_appointment", "medical_history_for_treatment"]
}}
""",
            
            "appointment_focused": """
You are a healthcare appointment planning specialist. Create a detailed plan for appointment-related queries.

User Query: {query}
Patient Context: {context}

Focus on:
1. Specialty identification
2. Urgency assessment
3. Patient preparation requirements
4. Follow-up needs

Return JSON plan for appointment booking workflow.
""",
            
            "medical_research": """
You are a medical information research planner. Create a plan for medical information queries.

User Query: {query}
Patient Context: {context}

Focus on:
1. Medical condition identification
2. Information type needed (treatment, diagnosis, prevention)
3. Source prioritization (guidelines, research, patient education)
4. Patient-specific considerations

Return JSON plan for medical research workflow.
"""
        }
    
    def create_detailed_plan(self, query: str, patient_context: Optional[Dict] = None) -> Dict:
        """Create detailed execution plan with proper goal decomposition"""
        try:
            # Determine query type and select appropriate template
            template_key = self._classify_query_type(query)
            template = self.planning_templates[template_key]
            
            # Format context
            context_str = json.dumps(patient_context, indent=2) if patient_context else "No prior context available"
            
            # Create planning prompt
            planning_prompt = template.format(query=query, context=context_str)
            
            # Get LLM response
            response = self.llm.invoke(planning_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract and validate JSON plan
            plan = self._extract_and_validate_plan(response_content, query)
            
            # Enhance plan with additional metadata
            plan = self._enhance_plan_metadata(plan, query)
            
            return plan
            
        except Exception as e:
            # Create intelligent fallback plan
            return self._create_intelligent_fallback_plan(query, str(e))
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type to select appropriate planning template"""
        query_lower = query.lower()
        
        # Multi-task queries (appointment + information)
        if (any(word in query_lower for word in ["book", "appointment", "schedule"]) and 
            any(word in query_lower for word in ["treatment", "information", "latest", "research"])):
            return "multi_task"
        
        # Appointment-focused queries
        elif any(word in query_lower for word in ["book", "appointment", "schedule", "doctor"]):
            return "appointment_focused"
        
        # Medical research queries
        elif any(word in query_lower for word in ["treatment", "information", "disease", "condition"]):
            return "medical_research"
        
        else:
            return "multi_task"  # Default to most comprehensive
    
    def _extract_and_validate_plan(self, response_content: str, original_query: str) -> Dict:
        """Extract and validate JSON plan from LLM response"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = ["patient_id", "intent", "tasks", "priority", "estimated_steps"]
                if all(field in plan for field in required_fields):
                    return plan
            
            # If extraction fails, create structured plan
            return self._create_structured_fallback_plan(original_query)
            
        except json.JSONDecodeError:
            return self._create_structured_fallback_plan(original_query)
    
    def _enhance_plan_metadata(self, plan: Dict, query: str) -> Dict:
        """Enhance plan with additional metadata and validation"""
        # Add query analysis
        plan["query_analysis"] = {
            "original_query": query,
            "complexity": self._assess_query_complexity(query),
            "medical_domains": self._identify_medical_domains(query),
            "urgency_indicators": self._identify_urgency_indicators(query)
        }
        
        # Add execution metadata
        plan["execution_metadata"] = {
            "created_at": "2024-12-22T08:30:00Z",
            "planner_version": "enhanced_v1.0",
            "confidence_score": self._calculate_confidence_score(plan)
        }
        
        # Validate and adjust task sequence
        plan["tasks"] = self._optimize_task_sequence(plan.get("tasks", []))
        
        return plan
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity for resource allocation"""
        query_lower = query.lower()
        
        # Count different types of requests
        complexity_indicators = 0
        
        if any(word in query_lower for word in ["book", "appointment", "schedule"]):
            complexity_indicators += 1
        
        if any(word in query_lower for word in ["history", "record", "patient"]):
            complexity_indicators += 1
        
        if any(word in query_lower for word in ["treatment", "information", "research"]):
            complexity_indicators += 1
        
        if any(word in query_lower for word in ["latest", "recent", "new"]):
            complexity_indicators += 1
        
        if complexity_indicators >= 3:
            return "high"
        elif complexity_indicators >= 2:
            return "medium"
        else:
            return "low"
    
    def _identify_medical_domains(self, query: str) -> List[str]:
        """Identify medical domains mentioned in query"""
        query_lower = query.lower()
        domains = []
        
        domain_keywords = {
            "nephrology": ["kidney", "nephrologist", "dialysis", "renal"],
            "cardiology": ["heart", "cardiologist", "cardiac", "cardiovascular"],
            "neurology": ["brain", "neurologist", "neurological", "migraine"],
            "oncology": ["cancer", "oncologist", "tumor", "chemotherapy"],
            "endocrinology": ["diabetes", "endocrinologist", "hormone", "thyroid"],
            "general_medicine": ["general", "primary", "family", "internal"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["general_medicine"]
    
    def _identify_urgency_indicators(self, query: str) -> List[str]:
        """Identify urgency indicators in query"""
        query_lower = query.lower()
        urgency_indicators = []
        
        urgent_keywords = ["urgent", "emergency", "asap", "immediately", "critical"]
        routine_keywords = ["routine", "regular", "checkup", "follow-up"]
        
        if any(keyword in query_lower for keyword in urgent_keywords):
            urgency_indicators.append("urgent")
        
        if any(keyword in query_lower for keyword in routine_keywords):
            urgency_indicators.append("routine")
        
        return urgency_indicators
    
    def _calculate_confidence_score(self, plan: Dict) -> float:
        """Calculate confidence score for the plan"""
        score = 0.5  # Base score
        
        # Increase confidence based on plan completeness
        if plan.get("patient_id") != "unknown":
            score += 0.1
        
        if len(plan.get("tasks", [])) >= 2:
            score += 0.1
        
        if plan.get("priority") in ["high", "medium", "low"]:
            score += 0.1
        
        if "required_tools" in plan:
            score += 0.1
        
        if "execution_order" in plan:
            score += 0.1
        
        return min(score, 1.0)
    
    def _optimize_task_sequence(self, tasks: List[str]) -> List[str]:
        """Optimize task execution sequence"""
        if not tasks:
            return ["process_general_query"]
        
        # Ensure context retrieval comes first if needed
        optimized_tasks = []
        
        # Always start with context retrieval for known patients
        if any("patient" in task or "context" in task for task in tasks):
            if "retrieve_patient_context" not in tasks:
                optimized_tasks.append("retrieve_patient_context")
        
        # Add other tasks in logical order
        task_priority = {
            "retrieve_patient_context": 1,
            "book_appointment": 2,
            "search_medical_information": 3,
            "update_medical_records": 4,
            "synthesize_response": 5
        }
        
        # Sort tasks by priority, keeping original order for same priority
        remaining_tasks = [task for task in tasks if task not in optimized_tasks]
        remaining_tasks.sort(key=lambda x: task_priority.get(x, 10))
        
        optimized_tasks.extend(remaining_tasks)
        
        return optimized_tasks
    
    def _create_structured_fallback_plan(self, query: str) -> Dict:
        """Create structured fallback plan when LLM parsing fails"""
        query_lower = query.lower()
        
        # Determine patient ID
        patient_id = "unknown"
        if "father" in query_lower:
            patient_id = "father"
        elif "mother" in query_lower:
            patient_id = "mother"
        elif any(name in query_lower for name in ["ramesh", "anjali", "david"]):
            for name in ["ramesh", "anjali", "david"]:
                if name in query_lower:
                    patient_id = name
                    break
        
        # Determine tasks
        tasks = ["retrieve_patient_context"] if patient_id != "unknown" else []
        
        if any(word in query_lower for word in ["book", "appointment", "schedule"]):
            tasks.append("book_appointment")
        
        if any(word in query_lower for word in ["history", "record", "show"]):
            tasks.append("retrieve_medical_records")
        
        if any(word in query_lower for word in ["treatment", "information", "latest", "search"]):
            tasks.append("search_medical_information")
        
        if not tasks or len(tasks) == 1:
            tasks.append("process_general_query")
        
        # Determine priority
        priority = "high" if any(word in query_lower for word in ["urgent", "emergency"]) else "medium"
        
        return {
            "patient_id": patient_id,
            "intent": "healthcare_assistance",
            "tasks": tasks,
            "priority": priority,
            "estimated_steps": len(tasks),
            "required_tools": ["memory", "appointment", "records", "search"],
            "execution_order": "sequential",
            "fallback_plan": True,
            "query_analysis": {
                "original_query": query,
                "complexity": self._assess_query_complexity(query),
                "medical_domains": self._identify_medical_domains(query)
            }
        }
    
    def _create_intelligent_fallback_plan(self, query: str, error: str) -> Dict:
        """Create intelligent fallback plan with error handling"""
        fallback_plan = self._create_structured_fallback_plan(query)
        fallback_plan["error"] = error
        fallback_plan["fallback_reason"] = "LLM planning failed, using rule-based fallback"
        return fallback_plan
