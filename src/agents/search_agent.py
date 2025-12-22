"""
Medical Search Agent - Retrieves medical information using RAG
"""

from typing import Dict, List, Any
import json

class MedicalSearchAgent:
    def __init__(self, llm, search_tools):
        self.llm = llm
        self.tools = search_tools
    
    def execute(self, plan: Dict) -> Dict:
        """Execute medical information search tasks"""
        try:
            tasks = plan.get("tasks", [])
            search_tasks = [t for t in tasks if any(keyword in t.lower() 
                          for keyword in ["search", "information", "treatment", "latest", "research"])]
            
            if not search_tasks:
                return {"status": "no_search_tasks", "message": "No search tasks found"}
            
            # Extract search query from tasks
            search_query = self._extract_search_query(search_tasks[0])
            
            # Perform medical information search
            search_results = self.tools.search_medical_info(search_query)
            
            if not search_results:
                return {
                    "status": "no_results",
                    "message": f"No medical information found for: {search_query}",
                    "query": search_query
                }
            
            # Generate comprehensive summary using RAG
            summary = self._generate_rag_summary(search_query, search_results)
            
            return {
                "status": "success",
                "query": search_query,
                "search_results": search_results,
                "summary": summary,
                "message": "Medical information retrieved and summarized successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Medical search failed: {str(e)}"
            }
    
    def _extract_search_query(self, task: str) -> str:
        """Extract search query from task description"""
        # Simple extraction - in production, use more sophisticated NLP
        keywords = ["treatment", "therapy", "medication", "diagnosis", "symptoms", "disease"]
        
        task_lower = task.lower()
        for keyword in keywords:
            if keyword in task_lower:
                # Extract context around the keyword
                words = task_lower.split()
                if keyword in words:
                    idx = words.index(keyword)
                    # Get surrounding context
                    start = max(0, idx - 2)
                    end = min(len(words), idx + 3)
                    return " ".join(words[start:end])
        
        return task
    
    def _generate_rag_summary(self, query: str, search_results: List[Dict]) -> str:
        """Generate RAG-based summary of search results"""
        try:
            # Combine search results into context
            context = "\n\n".join([
                f"Source: {result.get('source', 'Unknown')}\n{result.get('content', '')}"
                for result in search_results[:5]  # Top 5 results
            ])
            
            prompt = f"""
            Based on the following medical information sources, provide a comprehensive summary about: {query}
            
            Medical Sources:
            {context}
            
            Provide a summary that includes:
            1. Current understanding of the condition/treatment
            2. Latest treatment approaches
            3. Key considerations for patients
            4. Important warnings or contraindications
            
            Ensure the information is accurate and cite sources where appropriate.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"RAG summary generation failed: {str(e)}"
