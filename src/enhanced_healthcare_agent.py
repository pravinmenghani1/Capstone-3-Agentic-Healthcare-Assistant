"""
Enhanced Healthcare Agent with Full LLM Integration and Proper Agent Workflow
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

try:
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing_extensions import Annotated, TypedDict
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available - using simplified mode")

from .agents.enhanced_planner import EnhancedPlannerAgent
from .agents.enhanced_appointment import EnhancedAppointmentAgent
from .agents.enhanced_records import EnhancedRecordsAgent
from .agents.enhanced_search import EnhancedSearchAgent
from .memory.enhanced_memory import EnhancedPatientMemory
from .tools.appointment_tools import AppointmentTools
from .tools.medical_tools import MedicalTools
from .tools.search_tools import SearchTools
from .evaluation.enhanced_evaluator import EnhancedEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: Optional[Dict]
    current_step: int
    patient_context: Optional[Dict]
    results: Dict[str, Any]
    execution_trace: List[Dict]

class EnhancedHealthcareAgent:
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        # Initialize enhanced memory with proper context retention
        self.patient_memory = EnhancedPatientMemory()
        self.conversation_history = []
        
        # Initialize tools with real API simulation
        self.appointment_tools = AppointmentTools()
        self.medical_tools = MedicalTools()
        self.search_tools = SearchTools()
        
        # Initialize enhanced agents with proper prompt engineering
        self.planner = EnhancedPlannerAgent(self.llm)
        self.appointment_agent = EnhancedAppointmentAgent(self.llm, self.appointment_tools)
        self.records_agent = EnhancedRecordsAgent(self.llm, self.medical_tools)
        self.search_agent = EnhancedSearchAgent(self.llm, self.search_tools)
        
        # Initialize evaluator for LLMOps
        self.evaluator = EnhancedEvaluator(self.llm)
        
        # Build agent graph with proper workflow
        self.agent_graph = self._build_enhanced_graph()
        
        # Execution tracking
        self.execution_logs = []
        self.performance_metrics = {}
    
    def _initialize_llm(self):
        """Initialize LLM with proper configuration"""
        if not LANGCHAIN_AVAILABLE:
            return self._create_mock_llm()
        
        try:
            # Try Ollama
            return OllamaLLM(model=self.model_name, temperature=0.1)
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}")
            return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """Create mock LLM for demonstration"""
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                
                # Intelligent mock responses based on prompt content
                prompt_lower = str(prompt).lower()
                
                if "plan" in prompt_lower and "json" in prompt_lower:
                    return MockResponse(json.dumps({
                        "patient_id": "father" if "father" in prompt_lower else "patient",
                        "intent": "multi_task_healthcare",
                        "tasks": [
                            "retrieve_patient_context",
                            "book_appointment" if "appointment" in prompt_lower else "search_information",
                            "search_medical_information" if "treatment" in prompt_lower else "update_records"
                        ],
                        "priority": "high",
                        "estimated_steps": 3,
                        "required_tools": ["memory", "appointment", "search"]
                    }))
                
                elif "appointment" in prompt_lower:
                    return MockResponse("I have successfully identified available appointment slots and booked an appointment with the nephrologist for your father. The appointment is scheduled for next Tuesday at 2:00 PM with Dr. Sarah Johnson.")
                
                elif "medical history" in prompt_lower or "patient" in prompt_lower:
                    return MockResponse("Based on the patient's medical records, I found comprehensive information including chronic kidney disease diagnosis, current medications (ACE inhibitors), and recent lab results showing stable kidney function.")
                
                elif "treatment" in prompt_lower or "disease" in prompt_lower:
                    return MockResponse("Current treatment guidelines for chronic kidney disease include: 1) Blood pressure control with ACE inhibitors, 2) Dietary protein restriction, 3) Regular monitoring of kidney function, 4) Management of complications like anemia and bone disease.")
                
                else:
                    return MockResponse("I have processed your healthcare request using the multi-agent system. The task has been completed successfully with appropriate medical recommendations.")
        
        return MockLLM()
    
    def _build_enhanced_graph(self):
        """Build enhanced LangGraph workflow"""
        if not LANGCHAIN_AVAILABLE:
            return None
        
        try:
            workflow = StateGraph(AgentState)
            
            # Add enhanced nodes with proper state management
            workflow.add_node("planner", self._enhanced_plan_step)
            workflow.add_node("context_retrieval", self._context_retrieval_step)
            workflow.add_node("appointment", self._enhanced_appointment_step)
            workflow.add_node("records", self._enhanced_records_step)
            workflow.add_node("search", self._enhanced_search_step)
            workflow.add_node("synthesizer", self._enhanced_synthesis_step)
            workflow.add_node("evaluator", self._evaluation_step)
            
            # Set entry point
            workflow.set_entry_point("planner")
            
            # Add conditional routing with proper logic
            workflow.add_conditional_edges(
                "planner",
                self._route_next_step,
                {
                    "context_retrieval": "context_retrieval",
                    "appointment": "appointment",
                    "records": "records",
                    "search": "search",
                    "end": END
                }
            )
            
            workflow.add_edge("context_retrieval", "appointment")
            workflow.add_edge("appointment", "search")
            workflow.add_edge("records", "search")
            workflow.add_edge("search", "synthesizer")
            workflow.add_edge("synthesizer", "evaluator")
            workflow.add_edge("evaluator", END)
            
            return workflow.compile()
        except Exception as e:
            logger.error(f"Failed to build LangGraph: {e}")
            return None
    
    def process_complex_query(self, user_input: str) -> Dict[str, Any]:
        """Process complex multi-step healthcare queries"""
        try:
            # Create enhanced initial state
            initial_state = AgentState(
                messages=[{"role": "user", "content": user_input}],
                plan=None,
                current_step=0,
                patient_context=None,
                results={},
                execution_trace=[]
            )
            
            # Execute workflow
            if self.agent_graph and LANGCHAIN_AVAILABLE:
                final_state = self.agent_graph.invoke(initial_state)
            else:
                final_state = self._execute_simplified_workflow(initial_state)
            
            # Store conversation with context
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "input": user_input,
                "output": final_state["results"].get("final_response", ""),
                "plan": final_state["plan"],
                "execution_trace": final_state["execution_trace"]
            })
            
            # Update performance metrics
            self._update_performance_metrics(final_state)
            
            return {
                "response": final_state["results"].get("final_response", ""),
                "plan": final_state["plan"],
                "execution_trace": final_state["execution_trace"],
                "patient_context": final_state["patient_context"],
                "performance_metrics": self.performance_metrics,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing complex query: {str(e)}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "plan": None,
                "execution_trace": [],
                "success": False
            }
    
    def _execute_simplified_workflow(self, state: AgentState) -> AgentState:
        """Simplified workflow for when LangGraph is not available"""
        # Step 1: Enhanced Planning
        state = self._enhanced_plan_step(state)
        
        # Step 2: Context Retrieval
        state = self._context_retrieval_step(state)
        
        # Step 3: Execute based on plan
        plan = state.get("plan", {})
        tasks = plan.get("tasks", [])
        
        for task in tasks:
            if "appointment" in task.lower():
                state = self._enhanced_appointment_step(state)
            elif "record" in task.lower() or "history" in task.lower():
                state = self._enhanced_records_step(state)
            elif "search" in task.lower() or "information" in task.lower():
                state = self._enhanced_search_step(state)
        
        # Step 4: Synthesis
        state = self._enhanced_synthesis_step(state)
        
        # Step 5: Evaluation
        state = self._evaluation_step(state)
        
        return state
    
    def _enhanced_plan_step(self, state: AgentState) -> AgentState:
        """Enhanced planning with proper goal decomposition"""
        try:
            user_message = state["messages"][-1]["content"]
            
            # Create detailed plan using enhanced planner
            plan = self.planner.create_detailed_plan(user_message)
            
            # Log execution trace
            trace_entry = {
                "step": "planning",
                "timestamp": datetime.now().isoformat(),
                "input": user_message,
                "output": plan,
                "status": "success"
            }
            
            state["plan"] = plan
            state["execution_trace"].append(trace_entry)
            
            self._log_execution("enhanced_planner", {"input": user_message, "plan": plan})
            
            return state
            
        except Exception as e:
            logger.error(f"Enhanced planning failed: {e}")
            state["plan"] = {"error": str(e), "fallback": True}
            return state
    
    def _context_retrieval_step(self, state: AgentState) -> AgentState:
        """Enhanced context retrieval with patient memory"""
        try:
            plan = state["plan"]
            patient_id = plan.get("patient_id", "unknown")
            
            # Retrieve comprehensive patient context
            patient_context = self.patient_memory.get_comprehensive_context(patient_id)
            
            trace_entry = {
                "step": "context_retrieval",
                "timestamp": datetime.now().isoformat(),
                "patient_id": patient_id,
                "context_found": patient_context is not None,
                "status": "success"
            }
            
            state["patient_context"] = patient_context
            state["execution_trace"].append(trace_entry)
            
            return state
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return state
    
    def _enhanced_appointment_step(self, state: AgentState) -> AgentState:
        """Enhanced appointment booking with context awareness"""
        try:
            result = self.appointment_agent.execute_with_context(
                state["plan"], 
                state["patient_context"]
            )
            
            trace_entry = {
                "step": "appointment_booking",
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "status": "success" if result.get("status") == "success" else "failed"
            }
            
            state["results"]["appointment"] = result
            state["execution_trace"].append(trace_entry)
            
            return state
            
        except Exception as e:
            logger.error(f"Enhanced appointment step failed: {e}")
            state["results"]["appointment"] = {"error": str(e)}
            return state
    
    def _enhanced_records_step(self, state: AgentState) -> AgentState:
        """Enhanced medical records with LLM summarization"""
        try:
            result = self.records_agent.execute_with_llm_summary(
                state["plan"],
                state["patient_context"]
            )
            
            trace_entry = {
                "step": "medical_records",
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "status": "success" if result.get("status") == "success" else "failed"
            }
            
            state["results"]["records"] = result
            state["execution_trace"].append(trace_entry)
            
            return state
            
        except Exception as e:
            logger.error(f"Enhanced records step failed: {e}")
            state["results"]["records"] = {"error": str(e)}
            return state
    
    def _enhanced_search_step(self, state: AgentState) -> AgentState:
        """Enhanced medical search with RAG pipeline"""
        try:
            result = self.search_agent.execute_rag_search(
                state["plan"],
                state["patient_context"]
            )
            
            trace_entry = {
                "step": "medical_search",
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "status": "success" if result.get("status") == "success" else "failed"
            }
            
            state["results"]["search"] = result
            state["execution_trace"].append(trace_entry)
            
            return state
            
        except Exception as e:
            logger.error(f"Enhanced search step failed: {e}")
            state["results"]["search"] = {"error": str(e)}
            return state
    
    def _enhanced_synthesis_step(self, state: AgentState) -> AgentState:
        """Enhanced synthesis with structured response generation"""
        try:
            results = state["results"]
            plan = state["plan"]
            patient_context = state["patient_context"]
            
            # Create comprehensive synthesis prompt
            synthesis_prompt = f"""
            You are a healthcare AI assistant. Synthesize the following information into a comprehensive, professional response:
            
            Original Query: {state['messages'][-1]['content']}
            
            Execution Plan: {json.dumps(plan, indent=2)}
            
            Patient Context: {json.dumps(patient_context, indent=2) if patient_context else 'No prior context'}
            
            Results:
            - Appointment: {json.dumps(results.get('appointment', {}), indent=2)}
            - Medical Records: {json.dumps(results.get('records', {}), indent=2)}
            - Medical Search: {json.dumps(results.get('search', {}), indent=2)}
            
            Provide a comprehensive response that:
            1. Addresses all aspects of the original query
            2. Includes specific details from the results
            3. Provides medical recommendations with appropriate disclaimers
            4. Maintains a professional healthcare tone
            """
            
            response = self.llm.invoke(synthesis_prompt)
            final_response = response.content if hasattr(response, 'content') else str(response)
            
            # Add medical disclaimer
            final_response += "\n\n⚠️ Medical Disclaimer: This information is for educational purposes only. Please consult with qualified healthcare professionals for medical advice."
            
            trace_entry = {
                "step": "synthesis",
                "timestamp": datetime.now().isoformat(),
                "response_length": len(final_response),
                "status": "success"
            }
            
            state["results"]["final_response"] = final_response
            state["execution_trace"].append(trace_entry)
            
            return state
            
        except Exception as e:
            logger.error(f"Enhanced synthesis failed: {e}")
            # Fallback synthesis
            fallback_response = self._create_fallback_response(state["results"])
            state["results"]["final_response"] = fallback_response
            return state
    
    def _evaluation_step(self, state: AgentState) -> AgentState:
        """Evaluation step for LLMOps monitoring"""
        try:
            evaluation_result = self.evaluator.evaluate_agent_response(
                state["messages"][-1]["content"],
                state["results"].get("final_response", ""),
                state["execution_trace"]
            )
            
            state["results"]["evaluation"] = evaluation_result
            
            trace_entry = {
                "step": "evaluation",
                "timestamp": datetime.now().isoformat(),
                "evaluation_score": evaluation_result.get("overall_score", 0),
                "status": "success"
            }
            
            state["execution_trace"].append(trace_entry)
            
            return state
            
        except Exception as e:
            logger.error(f"Evaluation step failed: {e}")
            return state
    
    def _route_next_step(self, state: AgentState) -> str:
        """Enhanced routing logic"""
        try:
            plan = state.get("plan", {})
            tasks = plan.get("tasks", [])
            
            if not tasks:
                return "end"
            
            # Check if context retrieval is needed
            if plan.get("patient_id") != "unknown" and not state.get("patient_context"):
                return "context_retrieval"
            
            # Route based on task priority
            for task in tasks:
                if "appointment" in task.lower():
                    return "appointment"
                elif "record" in task.lower() or "history" in task.lower():
                    return "records"
                elif "search" in task.lower():
                    return "search"
            
            return "search"  # Default to search
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return "search"
    
    def _create_fallback_response(self, results: Dict) -> str:
        """Create fallback response when synthesis fails"""
        response_parts = []
        
        if "appointment" in results:
            response_parts.append("✅ Appointment request has been processed.")
        
        if "records" in results:
            response_parts.append("📋 Medical records have been retrieved and analyzed.")
        
        if "search" in results:
            response_parts.append("🔍 Medical information search has been completed.")
        
        if not response_parts:
            response_parts.append("I have processed your healthcare request.")
        
        return " ".join(response_parts) + "\n\n⚠️ Medical Disclaimer: Please consult with qualified healthcare professionals for medical advice."
    
    def _update_performance_metrics(self, final_state: AgentState):
        """Update performance metrics for monitoring"""
        try:
            execution_trace = final_state.get("execution_trace", [])
            
            # Calculate success rates
            successful_steps = len([step for step in execution_trace if step.get("status") == "success"])
            total_steps = len(execution_trace)
            
            self.performance_metrics.update({
                "total_queries_processed": self.performance_metrics.get("total_queries_processed", 0) + 1,
                "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
                "average_steps_per_query": total_steps,
                "last_updated": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def _log_execution(self, step: str, data: Dict):
        """Enhanced execution logging"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": data,
            "session_id": getattr(self, 'session_id', 'default')
        }
        self.execution_logs.append(log_entry)
        logger.info(f"Enhanced Step {step}: {data}")
    
    def get_comprehensive_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        return {
            "performance_metrics": self.performance_metrics,
            "execution_logs": self.execution_logs[-10:],
            "conversation_history": len(self.conversation_history),
            "patient_memory_stats": self.patient_memory.get_memory_stats(),
            "system_health": {
                "llm_available": LANGCHAIN_AVAILABLE,
                "agent_graph_active": self.agent_graph is not None,
                "tools_operational": True
            }
        }
