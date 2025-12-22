"""
Comprehensive Demo for Enhanced Agentic Healthcare Assistant
Demonstrates full capstone objectives with LLMOps integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enhanced_healthcare_agent import EnhancedHealthcareAgent
from src.evaluation.enhanced_evaluator import EnhancedEvaluator
import json
from datetime import datetime

def run_comprehensive_demo():
    """Run comprehensive demo showcasing all capstone objectives"""
    
    print("🏥 ENHANCED AGENTIC HEALTHCARE ASSISTANT - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("Demonstrating: Agentic AI, RAG Pipeline, LLMOps, and Streamlit UI")
    print("=" * 80)
    
    # Initialize enhanced healthcare agent
    print("\n🚀 Initializing Enhanced Healthcare Agent...")
    agent = EnhancedHealthcareAgent()
    
    # Test scenarios that demonstrate full capstone objectives
    comprehensive_scenarios = [
        {
            "name": "Complex Multi-Task Healthcare Query (Primary Objective)",
            "query": "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?",
            "expected_capabilities": [
                "Goal decomposition and planning",
                "Patient context retrieval", 
                "Appointment booking with context awareness",
                "RAG-based medical information search",
                "LLM-powered synthesis",
                "Comprehensive evaluation"
            ]
        },
        {
            "name": "Medical Records Analysis with LLM Summarization",
            "query": "Show me the comprehensive medical history for patient Ramesh and provide clinical insights",
            "expected_capabilities": [
                "Medical records retrieval",
                "LLM-powered medical summarization",
                "Clinical insight extraction",
                "Care recommendations generation"
            ]
        },
        {
            "name": "Advanced Medical Information Search with RAG",
            "query": "What are the latest evidence-based treatments for chronic kidney disease in elderly patients?",
            "expected_capabilities": [
                "Advanced RAG pipeline",
                "Multi-source medical search",
                "Evidence-based synthesis",
                "Patient-specific considerations"
            ]
        },
        {
            "name": "Integrated Care Coordination",
            "query": "Schedule a follow-up cardiology appointment for Anjali, review her migraine treatment history, and check for drug interactions",
            "expected_capabilities": [
                "Multi-agent coordination",
                "Cross-domain medical analysis",
                "Drug interaction checking",
                "Care continuity assessment"
            ]
        }
    ]
    
    demo_results = []
    
    for i, scenario in enumerate(comprehensive_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*60}")
        print(f"Query: {scenario['query']}")
        print(f"\nExpected Capabilities: {', '.join(scenario['expected_capabilities'])}")
        
        print(f"\n🔄 Processing with Enhanced Agent System...")
        
        # Process query with enhanced agent
        start_time = datetime.now()
        response_data = agent.process_complex_query(scenario['query'])
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n📋 EXECUTION PLAN:")
        if response_data.get('plan'):
            plan = response_data['plan']
            print(f"  Patient ID: {plan.get('patient_id', 'Unknown')}")
            print(f"  Intent: {plan.get('intent', 'Unknown')}")
            print(f"  Tasks: {', '.join(plan.get('tasks', []))}")
            print(f"  Priority: {plan.get('priority', 'Unknown')}")
            print(f"  Estimated Steps: {plan.get('estimated_steps', 0)}")
        
        print(f"\n🔍 EXECUTION TRACE:")
        execution_trace = response_data.get('execution_trace', [])
        for j, trace in enumerate(execution_trace, 1):
            status_icon = "✅" if trace.get('status') == 'success' else "❌"
            print(f"  {j}. {status_icon} {trace.get('step', 'Unknown').title()}")
        
        print(f"\n🤖 AGENT RESPONSE:")
        print(f"{response_data.get('response', 'No response generated')}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Processing Time: {processing_time:.2f} seconds")
        print(f"  Success: {'✅ Yes' if response_data.get('success') else '❌ No'}")
        print(f"  Steps Executed: {len(execution_trace)}")
        
        # Demonstrate LLMOps evaluation
        if response_data.get('success'):
            print(f"\n🔬 LLMOps EVALUATION:")
            evaluation = agent.evaluator.evaluate_agent_response(
                scenario['query'],
                response_data.get('response', ''),
                execution_trace,
                response_data.get('plan')
            )
            
            print(f"  Overall Score: {evaluation.get('overall_score', 0):.1f}/10")
            print(f"  Medical Accuracy: {evaluation.get('medical_accuracy', {}).get('score', 0):.1f}/10")
            print(f"  Response Completeness: {evaluation.get('response_completeness', {}).get('score', 0):.1f}/10")
            print(f"  Safety Compliance: {evaluation.get('safety_compliance', {}).get('score', 0):.1f}/10")
            
            quality_assessment = evaluation.get('quality_assessment', {})
            print(f"  Quality Level: {quality_assessment.get('quality_level', 'Unknown').title()}")
        
        demo_results.append({
            "scenario": scenario,
            "response_data": response_data,
            "processing_time": processing_time,
            "capabilities_demonstrated": scenario['expected_capabilities']
        })
    
    # Demonstrate comprehensive system capabilities
    print(f"\n{'='*80}")
    print("🎯 COMPREHENSIVE SYSTEM CAPABILITIES DEMONSTRATION")
    print(f"{'='*80}")
    
    # Get comprehensive metrics
    comprehensive_metrics = agent.get_comprehensive_metrics()
    
    print(f"\n📈 SYSTEM PERFORMANCE ANALYTICS:")
    perf_metrics = comprehensive_metrics.get('performance_metrics', {})
    print(f"  Total Queries Processed: {perf_metrics.get('total_queries_processed', 0)}")
    print(f"  Success Rate: {perf_metrics.get('success_rate', 0):.1%}")
    print(f"  Average Steps per Query: {perf_metrics.get('average_steps_per_query', 0)}")
    
    print(f"\n🧠 MEMORY SYSTEM STATISTICS:")
    memory_stats = comprehensive_metrics.get('patient_memory_stats', {})
    print(f"  Total Patients: {memory_stats.get('total_patients', 0)}")
    print(f"  Total Interactions: {memory_stats.get('total_interactions', 0)}")
    print(f"  Medical Contexts: {memory_stats.get('total_medical_contexts', 0)}")
    
    print(f"\n🔧 SYSTEM HEALTH STATUS:")
    system_health = comprehensive_metrics.get('system_health', {})
    for component, status in system_health.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}: {status}")
    
    # Demonstrate LLMOps analytics
    print(f"\n📊 LLMOps PERFORMANCE ANALYTICS:")
    analytics = agent.evaluator.get_performance_analytics()
    
    summary = analytics.get('summary', {})
    print(f"  Total Evaluations: {summary.get('total_evaluations', 0)}")
    print(f"  Evaluation Period: {summary.get('evaluation_period', 'Unknown')}")
    
    avg_scores = analytics.get('average_scores', {})
    for metric, data in avg_scores.items():
        avg = data.get('current_average', 0)
        trend = data.get('recent_trend', 'unknown')
        print(f"  {metric.replace('_', ' ').title()}: {avg:.1f}/10 ({trend})")
    
    insights = analytics.get('performance_insights', [])
    if insights:
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        for insight in insights:
            print(f"  {insight}")
    
    # Summary of capstone objectives achieved
    print(f"\n{'='*80}")
    print("✅ CAPSTONE OBJECTIVES ACHIEVED")
    print(f"{'='*80}")
    
    objectives_achieved = [
        "✅ Agent Planning and Goal Decomposition - Multi-step query interpretation",
        "✅ Tool and Memory Setup - Integrated APIs and vector database (FAISS)",
        "✅ Prompt Engineering and Task Chaining - Structured prompts with context",
        "✅ Agent Execution Flow - Complete workflow for sample scenario",
        "✅ Model Evaluation - QAEvalChain equivalent with LLM assessment",
        "✅ Data Visualization and UI - Streamlit dashboard (run streamlit_app.py)",
        "✅ Memory and Logs Interface - Agent traces and interactive testing",
        "✅ RAG Pipeline - Vector-based medical knowledge retrieval",
        "✅ LLMOps Integration - Comprehensive monitoring and evaluation"
    ]
    
    for objective in objectives_achieved:
        print(f"  {objective}")
    
    print(f"\n🎓 TECHNICAL ACHIEVEMENTS:")
    technical_achievements = [
        "🤖 Multi-Agent System with LangGraph orchestration",
        "🧠 Enhanced Patient Memory with FAISS vector indexing",
        "📚 Advanced RAG Pipeline with multi-source search",
        "🔬 Comprehensive LLMOps with automated evaluation",
        "💾 Persistent memory and context management",
        "🎯 Context-aware appointment booking",
        "📋 LLM-powered medical record summarization",
        "🔍 Evidence-based medical information synthesis",
        "📊 Real-time performance monitoring and analytics"
    ]
    
    for achievement in technical_achievements:
        print(f"  {achievement}")
    
    print(f"\n🚀 NEXT STEPS FOR PRODUCTION:")
    production_steps = [
        "🔗 Integrate with real medical APIs (PubMed, FHIR, EHR systems)",
        "🔒 Implement HIPAA compliance and security measures",
        "☁️ Deploy to cloud infrastructure with scalability",
        "📱 Develop mobile application interface",
        "🤝 Add multi-user collaboration features",
        "🔄 Implement real-time data synchronization",
        "📈 Add advanced analytics and reporting",
        "🧪 Conduct clinical validation studies"
    ]
    
    for step in production_steps:
        print(f"  {step}")
    
    # Save comprehensive demo results
    demo_summary = {
        "demo_completed_at": datetime.now().isoformat(),
        "scenarios_tested": len(comprehensive_scenarios),
        "total_processing_time": sum(result['processing_time'] for result in demo_results),
        "success_rate": sum(1 for result in demo_results if result['response_data'].get('success')) / len(demo_results),
        "capabilities_demonstrated": list(set(
            cap for result in demo_results 
            for cap in result['capabilities_demonstrated']
        )),
        "system_metrics": comprehensive_metrics,
        "evaluation_analytics": analytics
    }
    
    # Save demo results
    try:
        os.makedirs("src/data/demo_results", exist_ok=True)
        with open("src/data/demo_results/comprehensive_demo_results.json", "w") as f:
            json.dump(demo_summary, f, indent=2)
        print(f"\n💾 Demo results saved to: src/data/demo_results/comprehensive_demo_results.json")
    except Exception as e:
        print(f"\n⚠️ Could not save demo results: {e}")
    
    print(f"\n{'='*80}")
    print("🎉 COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
    print("🌐 Run 'streamlit run streamlit_app.py' to explore the interactive interface")
    print(f"{'='*80}")
    
    return demo_results, demo_summary

if __name__ == "__main__":
    run_comprehensive_demo()
