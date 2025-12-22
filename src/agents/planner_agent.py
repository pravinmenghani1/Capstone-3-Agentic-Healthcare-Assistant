"""
Planner Agent - Decomposes complex queries into actionable tasks
"""

from typing import Dict, List, Any, Optional

class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def create_plan(self, query: str, patient_context: Optional[Dict] = None) -> Dict:
        """Create execution plan from user query"""
        try:
            context_str = str(patient_context) if patient_context else "No prior context"
            
            # Create planning prompt
            planning_prompt = f"""
            You are a healthcare planning agent. Analyze the user query and break it down into specific, actionable tasks.
            
            User Query: {query}
            Patient Context: {context_str}
            
            Identify what needs to be done and categorize tasks into:
            1. Appointment booking (if scheduling is needed)
            2. Medical records (if patient history access/update is needed)  
            3. Medical search (if disease/treatment information is needed)
            
            Return a JSON plan with:
            {{
                "patient_id": "extracted or inferred patient identifier",
                "intent": "primary user intent",
                "tasks": ["list of specific tasks"],
                "priority": "high/medium/low",
                "estimated_steps": number
            }}
            """
            
            response = self.llm.invoke(planning_prompt)
            
            # Extract JSON from response
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    plan = json.loads(json_match.group())
                except:
                    plan = self._create_fallback_plan(query)
            else:
                plan = self._create_fallback_plan(query)
            
            return plan
            
        except Exception as e:
            # Return basic plan on error
            return self._create_fallback_plan(query, str(e))
    
    def _create_fallback_plan(self, query: str, error: str = None) -> Dict:
        """Create a fallback plan based on query analysis"""
        query_lower = query.lower()
        
        # Determine patient ID
        patient_id = "unknown"
        if "father" in query_lower:
            patient_id = "father"
        elif "ramesh" in query_lower:
            patient_id = "ramesh"
        elif "anjali" in query_lower:
            patient_id = "anjali"
        elif "david" in query_lower:
            patient_id = "david"
        
        # Determine tasks based on keywords
        tasks = []
        if any(word in query_lower for word in ["book", "appointment", "schedule"]):
            tasks.append("book appointment")
        if any(word in query_lower for word in ["history", "record", "show", "patient"]):
            tasks.append("retrieve medical records")
        if any(word in query_lower for word in ["treatment", "information", "latest", "search"]):
            tasks.append("search medical information")
        
        if not tasks:
            tasks = ["process general query"]
        
        # Determine priority
        priority = "high" if "urgent" in query_lower else "medium"
        
        plan = {
            "patient_id": patient_id,
            "intent": "healthcare_assistance",
            "tasks": tasks,
            "priority": priority,
            "estimated_steps": len(tasks)
        }
        
        if error:
            plan["error"] = error
        
        return plan
