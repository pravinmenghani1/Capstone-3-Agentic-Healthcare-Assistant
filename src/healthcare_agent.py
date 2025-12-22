"""
Core Healthcare Agent with Agentic AI capabilities
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

try:
    from langchain_community.llms import Ollama
except ImportError:
    print("Warning: Some LangChain modules not available. Using mock implementations.")
    
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing_extensions import Annotated, TypedDict
except ImportError:
    print("Warning: LangGraph not available. Using simplified workflow.")
    StateGraph = None
    END = "END"

from .agents.planner_agent import PlannerAgent
from .agents.appointment_agent import AppointmentAgent
from .agents.medical_records_agent import MedicalRecordsAgent
from .agents.search_agent import MedicalSearchAgent
from .memory.patient_memory import PatientMemory
from .tools.appointment_tools import AppointmentTools
from .tools.medical_tools import MedicalTools
from .tools.search_tools import SearchTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple state class for when LangGraph is not available
class AgentState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'messages' not in self:
            self['messages'] = []
        if 'plan' not in self:
            self['plan'] = None
        if 'current_step' not in self:
            self['current_step'] = 0
        if 'patient_context' not in self:
            self['patient_context'] = None
        if 'results' not in self:
            self['results'] = {}

class MockLLM:
    """Mock LLM for when real LLM is not available"""
    def invoke(self, prompt):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        # Simple mock responses based on prompt content
        if "plan" in prompt.lower():
            return MockResponse('{"patient_id": "patient", "intent": "general_query", "tasks": ["process request"], "priority": "medium", "estimated_steps": 1}')
        else:
            return MockResponse("This is a mock response for demonstration purposes.")

class HealthcareAgent:
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        # Initialize memory
        self.patient_memory = PatientMemory()
        self.conversation_memory = []  # Simple list instead of LangChain memory
        
        # Initialize tools
        self.appointment_tools = AppointmentTools()
        self.medical_tools = MedicalTools()
        self.search_tools = SearchTools()
        
        # Initialize specialized agents
        self.planner = PlannerAgent(self.llm)
        self.appointment_agent = AppointmentAgent(self.llm, self.appointment_tools)
        self.records_agent = MedicalRecordsAgent(self.llm, self.medical_tools)
        self.search_agent = MedicalSearchAgent(self.llm, self.search_tools)
        
        # Build the agent graph
        self.agent_graph = self._build_agent_graph()
        
        # Execution logs
        self.execution_logs = []
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            # Try Ollama
            return Ollama(model=self.model_name, temperature=0.1)
        except:
            # Use mock LLM
            print("Warning: Using mock LLM. Install langchain packages for full functionality.")
            return MockLLM()
    
    def _build_agent_graph(self):
        """Build the agent workflow (simplified when LangGraph not available)"""
        if StateGraph is None:
            return None  # Use simplified workflow
        
        try:
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("planner", self._plan_step)
            workflow.add_node("appointment", self._appointment_step)
            workflow.add_node("records", self._records_step)
            workflow.add_node("search", self._search_step)
            workflow.add_node("synthesizer", self._synthesize_step)
            
            # Set entry point
            workflow.set_entry_point("planner")
            
            # Add conditional edges
            workflow.add_conditional_edges(
                "planner",
                self._route_next_step,
                {
                    "appointment": "appointment",
                    "records": "records", 
                    "search": "search",
                    "end": END
                }
            )
            
            workflow.add_edge("appointment", "synthesizer")
            workflow.add_edge("records", "synthesizer")
            workflow.add_edge("search", "synthesizer")
            workflow.add_edge("synthesizer", END)
            
            return workflow.compile()
        except Exception as e:
            print(f"Warning: Could not build LangGraph workflow: {e}")
            return None
    
    def _plan_step(self, state: AgentState) -> AgentState:
        """Planning step - decompose user query into tasks"""
        try:
            user_message = state["messages"][-1]["content"] if state["messages"] else ""
            
            # Get patient context if available
            patient_context = self.patient_memory.get_patient_context(user_message)
            
            # Create plan
            plan = self.planner.create_plan(user_message, patient_context)
            
            self._log_execution("planner", {"input": user_message, "plan": plan})
            
            state["plan"] = plan
            state["current_step"] = 0
            state["patient_context"] = patient_context
            state["results"] = {}
            
            return state
        except Exception as e:
            logger.error(f"Error in plan step: {e}")
            state["plan"] = {"error": str(e)}
            return state
    
    def _appointment_step(self, state: AgentState) -> AgentState:
        """Handle appointment booking tasks"""
        try:
            plan = state["plan"]
            patient_context = state["patient_context"]
            
            result = self.appointment_agent.execute(plan, patient_context)
            
            self._log_execution("appointment", {"plan": plan, "result": result})
            
            state["results"]["appointment"] = result
            return state
        except Exception as e:
            logger.error(f"Error in appointment step: {e}")
            state["results"]["appointment"] = {"error": str(e)}
            return state
    
    def _records_step(self, state: AgentState) -> AgentState:
        """Handle medical records tasks"""
        try:
            plan = state["plan"]
            patient_context = state["patient_context"]
            
            result = self.records_agent.execute(plan, patient_context)
            
            self._log_execution("records", {"plan": plan, "result": result})
            
            state["results"]["records"] = result
            return state
        except Exception as e:
            logger.error(f"Error in records step: {e}")
            state["results"]["records"] = {"error": str(e)}
            return state
    
    def _search_step(self, state: AgentState) -> AgentState:
        """Handle medical information search tasks"""
        try:
            plan = state["plan"]
            
            result = self.search_agent.execute(plan)
            
            self._log_execution("search", {"plan": plan, "result": result})
            
            state["results"]["search"] = result
            return state
        except Exception as e:
            logger.error(f"Error in search step: {e}")
            state["results"]["search"] = {"error": str(e)}
            return state
    
    def _synthesize_step(self, state: AgentState) -> AgentState:
        """Synthesize results from all agents"""
        try:
            results = state["results"]
            plan = state["plan"]
            
            # Create synthesis prompt
            synthesis_prompt = f"""
            Based on the following plan and results, provide a comprehensive response:
            
            Plan: {plan}
            Results: {json.dumps(results, indent=2)}
            
            Synthesize the information into a helpful, coherent response for the user.
            """
            
            response = self.llm.invoke(synthesis_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            self._log_execution("synthesizer", {"results": results, "response": response_content})
            
            state["results"]["final_response"] = response_content
            return state
        except Exception as e:
            logger.error(f"Error in synthesize step: {e}")
            # Create a basic response from available results
            response = "I've processed your request. "
            if "appointment" in state["results"]:
                response += "Appointment information has been handled. "
            if "records" in state["results"]:
                response += "Medical records have been retrieved. "
            if "search" in state["results"]:
                response += "Medical information has been searched. "
            
            state["results"]["final_response"] = response
            return state
    
    def _route_next_step(self, state: AgentState) -> str:
        """Route to the next step based on the plan"""
        try:
            plan = state.get("plan")
            
            if not plan or "tasks" not in plan:
                return "end"
            
            tasks = plan["tasks"]
            
            # Determine which agent to use based on task types
            if any("appointment" in str(task).lower() or "book" in str(task).lower() for task in tasks):
                return "appointment"
            elif any("record" in str(task).lower() or "history" in str(task).lower() for task in tasks):
                return "records"
            elif any("search" in str(task).lower() or "information" in str(task).lower() or "treatment" in str(task).lower() for task in tasks):
                return "search"
            else:
                return "search"  # Default to search for general queries
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            return "search"  # Default fallback
    
    def _log_execution(self, step: str, data: Dict):
        """Log execution details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": data
        }
        self.execution_logs.append(log_entry)
        logger.info(f"Step {step}: {data}")
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process a user query through the agent system"""
        try:
            # Create initial state
            initial_state = AgentState({
                "messages": [{"role": "user", "content": user_input}],
                "plan": None,
                "current_step": 0,
                "patient_context": None,
                "results": {}
            })
            
            # Use LangGraph if available, otherwise use simplified workflow
            if self.agent_graph:
                final_state = self.agent_graph.invoke(initial_state)
            else:
                final_state = self._simplified_workflow(initial_state)
            
            # Store conversation in memory
            self.conversation_memory.append({
                "input": user_input,
                "output": final_state["results"].get("final_response", "")
            })
            
            return {
                "response": final_state["results"].get("final_response", ""),
                "plan": final_state["plan"],
                "execution_logs": self.execution_logs[-10:],  # Last 10 logs
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "plan": None,
                "execution_logs": [],
                "success": False
            }
    
    def _simplified_workflow(self, state: AgentState) -> AgentState:
        """Simplified workflow when LangGraph is not available"""
        # Step 1: Planning
        state = self._plan_step(state)
        
        # Step 2: Route to appropriate agent
        next_step = self._route_next_step(state)
        
        if next_step == "appointment":
            state = self._appointment_step(state)
        elif next_step == "records":
            state = self._records_step(state)
        elif next_step == "search":
            state = self._search_step(state)
        
        # Step 3: Synthesize results
        state = self._synthesize_step(state)
        
        return state
    
    def get_patient_summary(self, patient_id: str) -> Dict:
        """Get patient summary from memory"""
        return self.patient_memory.get_patient_summary(patient_id)
    
    def get_execution_metrics(self) -> Dict:
        """Get execution metrics for monitoring"""
        if not self.execution_logs:
            return {"total_executions": 0}
        
        step_counts = {}
        success_rate = 0
        
        for log in self.execution_logs:
            step = log["step"]
            step_counts[step] = step_counts.get(step, 0) + 1
        
        return {
            "total_executions": len(self.execution_logs),
            "step_distribution": step_counts,
            "recent_logs": self.execution_logs[-5:]
        }
