"""
Enhanced Patient Memory with Advanced Context Management and Long-term Retention
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EnhancedPatientMemory:
    def __init__(self, memory_dir: str = "src/data/enhanced_memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize enhanced sentence transformer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize multiple FAISS indices for different data types
        self.dimension = 384
        self.patient_index = faiss.IndexFlatIP(self.dimension)
        self.interaction_index = faiss.IndexFlatIP(self.dimension)
        self.medical_context_index = faiss.IndexFlatIP(self.dimension)
        
        # Enhanced storage structures
        self.patient_profiles = {}
        self.interaction_history = {}
        self.medical_contexts = {}
        self.temporal_memory = {}
        
        # Memory metadata
        self.memory_metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_patients": 0,
            "total_interactions": 0
        }
        
        # Load existing enhanced memory
        self._load_enhanced_memory()
    
    def _load_enhanced_memory(self):
        """Load enhanced memory structures from disk"""
        try:
            # Load patient profiles
            profiles_file = os.path.join(self.memory_dir, "patient_profiles.json")
            if os.path.exists(profiles_file):
                with open(profiles_file, 'r') as f:
                    self.patient_profiles = json.load(f)
            
            # Load interaction history
            interactions_file = os.path.join(self.memory_dir, "interaction_history.json")
            if os.path.exists(interactions_file):
                with open(interactions_file, 'r') as f:
                    self.interaction_history = json.load(f)
            
            # Load medical contexts
            contexts_file = os.path.join(self.memory_dir, "medical_contexts.json")
            if os.path.exists(contexts_file):
                with open(contexts_file, 'r') as f:
                    self.medical_contexts = json.load(f)
            
            # Load metadata
            metadata_file = os.path.join(self.memory_dir, "memory_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.memory_metadata = json.load(f)
            
            # Rebuild FAISS indices
            self._rebuild_all_indices()
            
        except Exception as e:
            print(f"Error loading enhanced memory: {e}")
    
    def _save_enhanced_memory(self):
        """Save enhanced memory structures to disk"""
        try:
            # Save patient profiles
            profiles_file = os.path.join(self.memory_dir, "patient_profiles.json")
            with open(profiles_file, 'w') as f:
                json.dump(self.patient_profiles, f, indent=2)
            
            # Save interaction history
            interactions_file = os.path.join(self.memory_dir, "interaction_history.json")
            with open(interactions_file, 'w') as f:
                json.dump(self.interaction_history, f, indent=2)
            
            # Save medical contexts
            contexts_file = os.path.join(self.memory_dir, "medical_contexts.json")
            with open(contexts_file, 'w') as f:
                json.dump(self.medical_contexts, f, indent=2)
            
            # Update and save metadata
            self.memory_metadata["last_updated"] = datetime.now().isoformat()
            self.memory_metadata["total_patients"] = len(self.patient_profiles)
            self.memory_metadata["total_interactions"] = sum(len(interactions) for interactions in self.interaction_history.values())
            
            metadata_file = os.path.join(self.memory_dir, "memory_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.memory_metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving enhanced memory: {e}")
    
    def store_comprehensive_patient_context(self, patient_id: str, context_data: Dict):
        """Store comprehensive patient context with enhanced indexing"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Create or update patient profile
            if patient_id not in self.patient_profiles:
                self.patient_profiles[patient_id] = {
                    "patient_id": patient_id,
                    "created_at": timestamp,
                    "profile_version": "enhanced_v1.0",
                    "demographic_info": {},
                    "medical_summary": {},
                    "care_preferences": {},
                    "risk_factors": [],
                    "care_team": []
                }
            
            # Update patient profile with new context
            profile = self.patient_profiles[patient_id]
            
            # Merge demographic information
            if "demographic_info" in context_data:
                profile["demographic_info"].update(context_data["demographic_info"])
            
            # Update medical summary
            if "medical_summary" in context_data:
                profile["medical_summary"].update(context_data["medical_summary"])
            
            # Add to medical contexts with temporal indexing
            context_id = f"{patient_id}_{timestamp}"
            self.medical_contexts[context_id] = {
                "patient_id": patient_id,
                "timestamp": timestamp,
                "context_type": context_data.get("context_type", "general"),
                "data": context_data,
                "embedding_text": self._create_enhanced_embedding_text(context_data)
            }
            
            # Update temporal memory for time-based retrieval
            date_key = datetime.now().strftime("%Y-%m-%d")
            if date_key not in self.temporal_memory:
                self.temporal_memory[date_key] = []
            
            self.temporal_memory[date_key].append({
                "patient_id": patient_id,
                "context_id": context_id,
                "timestamp": timestamp
            })
            
            # Update FAISS indices
            self._update_indices_for_patient(patient_id, context_data)
            
            # Save to disk
            self._save_enhanced_memory()
            
        except Exception as e:
            print(f"Error storing comprehensive patient context: {e}")
    
    def get_comprehensive_context(self, patient_id: str, context_window_days: int = 30) -> Optional[Dict]:
        """Retrieve comprehensive patient context with temporal awareness"""
        try:
            if patient_id not in self.patient_profiles:
                return None
            
            # Get base patient profile
            patient_profile = self.patient_profiles[patient_id].copy()
            
            # Get recent interactions within context window
            cutoff_date = datetime.now() - timedelta(days=context_window_days)
            recent_interactions = []
            
            if patient_id in self.interaction_history:
                for interaction in self.interaction_history[patient_id]:
                    interaction_date = datetime.fromisoformat(interaction["timestamp"])
                    if interaction_date >= cutoff_date:
                        recent_interactions.append(interaction)
            
            # Get relevant medical contexts
            relevant_contexts = []
            for context_id, context in self.medical_contexts.items():
                if (context["patient_id"] == patient_id and 
                    datetime.fromisoformat(context["timestamp"]) >= cutoff_date):
                    relevant_contexts.append(context)
            
            # Compile comprehensive context
            comprehensive_context = {
                "patient_profile": patient_profile,
                "recent_interactions": recent_interactions,
                "medical_contexts": relevant_contexts,
                "context_summary": self._generate_context_summary(patient_profile, recent_interactions),
                "risk_assessment": self._assess_patient_risks(patient_profile, recent_interactions),
                "care_continuity": self._analyze_care_continuity(patient_id, recent_interactions),
                "retrieved_at": datetime.now().isoformat(),
                "context_window_days": context_window_days
            }
            
            return comprehensive_context
            
        except Exception as e:
            print(f"Error retrieving comprehensive context: {e}")
            return None
    
    def semantic_search_patient_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search across all patient contexts"""
        try:
            if not self.medical_contexts:
                return []
            
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Search medical contexts
            if self.medical_context_index.ntotal > 0:
                scores, indices = self.medical_context_index.search(
                    query_embedding.astype('float32'), 
                    min(top_k, self.medical_context_index.ntotal)
                )
                
                # Retrieve matching contexts
                matching_contexts = []
                context_list = list(self.medical_contexts.values())
                
                for score, idx in zip(scores[0], indices[0]):
                    if score > 0.3 and idx < len(context_list):  # Similarity threshold
                        context = context_list[idx].copy()
                        context["similarity_score"] = float(score)
                        matching_contexts.append(context)
                
                return matching_contexts
            
            return []
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def add_interaction_with_context(self, patient_id: str, interaction_data: Dict):
        """Add interaction with enhanced context tracking"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Initialize interaction history for patient if needed
            if patient_id not in self.interaction_history:
                self.interaction_history[patient_id] = []
            
            # Create enhanced interaction record
            enhanced_interaction = {
                "interaction_id": f"int_{len(self.interaction_history[patient_id]) + 1:04d}",
                "timestamp": timestamp,
                "interaction_type": interaction_data.get("type", "general"),
                "query": interaction_data.get("query", ""),
                "response": interaction_data.get("response", ""),
                "agent_actions": interaction_data.get("agent_actions", []),
                "execution_trace": interaction_data.get("execution_trace", []),
                "satisfaction_score": interaction_data.get("satisfaction_score"),
                "follow_up_needed": interaction_data.get("follow_up_needed", False),
                "embedding_text": self._create_interaction_embedding_text(interaction_data)
            }
            
            # Add to interaction history
            self.interaction_history[patient_id].append(enhanced_interaction)
            
            # Update interaction index
            self._update_interaction_index(enhanced_interaction)
            
            # Analyze interaction patterns
            self._analyze_interaction_patterns(patient_id)
            
            # Save to disk
            self._save_enhanced_memory()
            
        except Exception as e:
            print(f"Error adding interaction with context: {e}")
    
    def _create_enhanced_embedding_text(self, context_data: Dict) -> str:
        """Create enhanced embedding text from context data"""
        text_parts = []
        
        # Add demographic information
        if "demographic_info" in context_data:
            demo = context_data["demographic_info"]
            if "age" in demo:
                text_parts.append(f"age {demo['age']}")
            if "gender" in demo:
                text_parts.append(f"gender {demo['gender']}")
        
        # Add medical information
        if "medical_summary" in context_data:
            medical = context_data["medical_summary"]
            if "conditions" in medical:
                conditions = " ".join(medical["conditions"])
                text_parts.append(f"conditions {conditions}")
            if "medications" in medical:
                medications = " ".join(medical["medications"])
                text_parts.append(f"medications {medications}")
        
        # Add contextual information
        if "context_type" in context_data:
            text_parts.append(f"context {context_data['context_type']}")
        
        return " | ".join(text_parts)
    
    def _create_interaction_embedding_text(self, interaction_data: Dict) -> str:
        """Create embedding text from interaction data"""
        text_parts = []
        
        if "query" in interaction_data:
            text_parts.append(f"query: {interaction_data['query']}")
        
        if "type" in interaction_data:
            text_parts.append(f"type: {interaction_data['type']}")
        
        if "agent_actions" in interaction_data:
            actions = " ".join(interaction_data["agent_actions"])
            text_parts.append(f"actions: {actions}")
        
        return " | ".join(text_parts)
    
    def _update_indices_for_patient(self, patient_id: str, context_data: Dict):
        """Update FAISS indices for patient data"""
        try:
            # Update medical context index
            embedding_text = self._create_enhanced_embedding_text(context_data)
            embedding = self.encoder.encode([embedding_text])
            self.medical_context_index.add(embedding.astype('float32'))
            
        except Exception as e:
            print(f"Error updating indices: {e}")
    
    def _update_interaction_index(self, interaction: Dict):
        """Update interaction index"""
        try:
            embedding_text = interaction["embedding_text"]
            embedding = self.encoder.encode([embedding_text])
            self.interaction_index.add(embedding.astype('float32'))
            
        except Exception as e:
            print(f"Error updating interaction index: {e}")
    
    def _rebuild_all_indices(self):
        """Rebuild all FAISS indices from stored data"""
        try:
            # Rebuild medical context index
            if self.medical_contexts:
                embeddings = []
                for context in self.medical_contexts.values():
                    embedding_text = context.get("embedding_text", "")
                    if embedding_text:
                        embedding = self.encoder.encode([embedding_text])
                        embeddings.append(embedding[0])
                
                if embeddings:
                    embeddings_array = np.array(embeddings).astype('float32')
                    self.medical_context_index.add(embeddings_array)
            
            # Rebuild interaction index
            if self.interaction_history:
                embeddings = []
                for patient_interactions in self.interaction_history.values():
                    for interaction in patient_interactions:
                        embedding_text = interaction.get("embedding_text", "")
                        if embedding_text:
                            embedding = self.encoder.encode([embedding_text])
                            embeddings.append(embedding[0])
                
                if embeddings:
                    embeddings_array = np.array(embeddings).astype('float32')
                    self.interaction_index.add(embeddings_array)
            
        except Exception as e:
            print(f"Error rebuilding indices: {e}")
    
    def _generate_context_summary(self, patient_profile: Dict, recent_interactions: List[Dict]) -> str:
        """Generate context summary for patient"""
        summary_parts = []
        
        # Patient demographics
        demo = patient_profile.get("demographic_info", {})
        if demo:
            summary_parts.append(f"Patient: {demo.get('age', 'Unknown age')}, {demo.get('gender', 'Unknown gender')}")
        
        # Medical conditions
        medical = patient_profile.get("medical_summary", {})
        conditions = medical.get("conditions", [])
        if conditions:
            summary_parts.append(f"Conditions: {', '.join(conditions)}")
        
        # Recent activity
        if recent_interactions:
            summary_parts.append(f"Recent interactions: {len(recent_interactions)} in last 30 days")
        
        return " | ".join(summary_parts) if summary_parts else "Limited context available"
    
    def _assess_patient_risks(self, patient_profile: Dict, recent_interactions: List[Dict]) -> Dict:
        """Assess patient risks based on profile and interactions"""
        risks = {
            "clinical_risks": [],
            "care_continuity_risks": [],
            "medication_risks": [],
            "overall_risk_level": "low"
        }
        
        # Analyze medical conditions for risks
        medical = patient_profile.get("medical_summary", {})
        conditions = medical.get("conditions", [])
        
        for condition in conditions:
            condition_lower = condition.lower()
            if any(high_risk in condition_lower for high_risk in ["diabetes", "hypertension", "kidney", "heart"]):
                risks["clinical_risks"].append(f"High-risk condition: {condition}")
        
        # Analyze interaction patterns
        if len(recent_interactions) > 5:
            risks["care_continuity_risks"].append("High interaction frequency - monitor for care fragmentation")
        elif len(recent_interactions) == 0:
            risks["care_continuity_risks"].append("No recent interactions - potential care gap")
        
        # Determine overall risk level
        total_risks = len(risks["clinical_risks"]) + len(risks["care_continuity_risks"]) + len(risks["medication_risks"])
        if total_risks >= 3:
            risks["overall_risk_level"] = "high"
        elif total_risks >= 1:
            risks["overall_risk_level"] = "moderate"
        
        return risks
    
    def _analyze_care_continuity(self, patient_id: str, recent_interactions: List[Dict]) -> Dict:
        """Analyze care continuity for patient"""
        continuity = {
            "interaction_frequency": len(recent_interactions),
            "care_gaps": [],
            "follow_up_compliance": "unknown",
            "care_coordination_score": 0.5
        }
        
        if recent_interactions:
            # Analyze follow-up compliance
            follow_up_needed = sum(1 for interaction in recent_interactions 
                                 if interaction.get("follow_up_needed", False))
            
            if follow_up_needed > 0:
                continuity["follow_up_compliance"] = "needs_attention"
            else:
                continuity["follow_up_compliance"] = "good"
            
            # Calculate care coordination score
            interaction_types = set(interaction.get("interaction_type", "general") 
                                  for interaction in recent_interactions)
            continuity["care_coordination_score"] = min(len(interaction_types) / 3.0, 1.0)
        
        return continuity
    
    def _analyze_interaction_patterns(self, patient_id: str):
        """Analyze interaction patterns for insights"""
        try:
            if patient_id not in self.interaction_history:
                return
            
            interactions = self.interaction_history[patient_id]
            
            # Analyze frequency patterns
            if len(interactions) >= 3:
                # Check for increasing frequency (potential escalation)
                recent_timestamps = [datetime.fromisoformat(i["timestamp"]) for i in interactions[-3:]]
                time_gaps = [(recent_timestamps[i+1] - recent_timestamps[i]).days 
                           for i in range(len(recent_timestamps)-1)]
                
                if all(gap <= 7 for gap in time_gaps):  # All within a week
                    # Flag for potential care escalation
                    pass
        
        except Exception as e:
            print(f"Error analyzing interaction patterns: {e}")
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        return {
            "total_patients": len(self.patient_profiles),
            "total_interactions": sum(len(interactions) for interactions in self.interaction_history.values()),
            "total_medical_contexts": len(self.medical_contexts),
            "memory_indices": {
                "patient_index_size": self.patient_index.ntotal,
                "interaction_index_size": self.interaction_index.ntotal,
                "medical_context_index_size": self.medical_context_index.ntotal
            },
            "temporal_coverage_days": len(self.temporal_memory),
            "last_updated": self.memory_metadata.get("last_updated", "Unknown")
        }
