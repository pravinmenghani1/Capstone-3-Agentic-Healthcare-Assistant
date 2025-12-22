"""
Patient Memory System - Vector-based patient context storage and retrieval
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class PatientMemory:
    def __init__(self, memory_dir: str = "src/data/memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize sentence transformer with error handling
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: SentenceTransformer failed ({e}), using fallback")
            self.encoder = None
        
        # Initialize FAISS index
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        if self.encoder is not None:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = None
        
        # Patient data storage
        self.patient_data = {}
        self.patient_embeddings = []
        self.patient_ids = []
        
        # Load existing memory
        self._load_memory()
    
    def _load_memory(self):
        """Load existing patient memory from disk"""
        try:
            memory_file = os.path.join(self.memory_dir, "patient_memory.json")
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    self.patient_data = json.load(f)
                
                # Rebuild FAISS index
                if self.patient_data:
                    texts = []
                    ids = []
                    for patient_id, data in self.patient_data.items():
                        # Create searchable text from patient data
                        text = self._create_searchable_text(data)
                        texts.append(text)
                        ids.append(patient_id)
                    
                    if texts:
                        embeddings = self.encoder.encode(texts)
                        self.index.add(embeddings.astype('float32'))
                        self.patient_embeddings = embeddings
                        self.patient_ids = ids
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def _save_memory(self):
        """Save patient memory to disk"""
        try:
            memory_file = os.path.join(self.memory_dir, "patient_memory.json")
            with open(memory_file, 'w') as f:
                json.dump(self.patient_data, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def _create_searchable_text(self, patient_data: Dict) -> str:
        """Create searchable text representation of patient data"""
        text_parts = []
        
        # Basic info
        if 'name' in patient_data:
            text_parts.append(f"Patient: {patient_data['name']}")
        
        # Medical conditions
        if 'conditions' in patient_data:
            conditions = ", ".join(patient_data['conditions'])
            text_parts.append(f"Conditions: {conditions}")
        
        # Recent visits
        if 'recent_visits' in patient_data:
            for visit in patient_data['recent_visits'][-3:]:  # Last 3 visits
                text_parts.append(f"Visit: {visit.get('date', '')} - {visit.get('diagnosis', '')}")
        
        return " | ".join(text_parts)
    
    def store_patient_context(self, patient_id: str, context: Dict):
        """Store patient context in memory"""
        try:
            # Update patient data
            if patient_id not in self.patient_data:
                self.patient_data[patient_id] = {
                    "created_at": datetime.now().isoformat(),
                    "interactions": []
                }
            
            # Merge new context
            for key, value in context.items():
                self.patient_data[patient_id][key] = value
            
            # Add interaction timestamp
            self.patient_data[patient_id]["last_updated"] = datetime.now().isoformat()
            
            # Update FAISS index
            searchable_text = self._create_searchable_text(self.patient_data[patient_id])
            embedding = self.encoder.encode([searchable_text])
            
            # Check if patient already exists in index
            if patient_id in self.patient_ids:
                # Update existing entry (rebuild index for simplicity)
                self._rebuild_index()
            else:
                # Add new entry
                self.index.add(embedding.astype('float32'))
                self.patient_embeddings.append(embedding[0])
                self.patient_ids.append(patient_id)
            
            self._save_memory()
            
        except Exception as e:
            print(f"Error storing patient context: {e}")
    
    def get_patient_context(self, query: str, top_k: int = 3) -> Optional[Dict]:
        """Retrieve relevant patient context based on query"""
        try:
            if not self.patient_ids or self.encoder is None or self.index is None:
                return None
            
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Search similar patients
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Get most relevant patient
            if len(indices[0]) > 0 and scores[0][0] > 0.3:  # Similarity threshold
                best_match_idx = indices[0][0]
                patient_id = self.patient_ids[best_match_idx]
                
                return {
                    "patient_id": patient_id,
                    "data": self.patient_data[patient_id],
                    "similarity_score": float(scores[0][0])
                }
            
            return None
            
        except Exception as e:
            print(f"Error retrieving patient context: {e}")
            return None
    
    def get_patient_summary(self, patient_id: str) -> Dict:
        """Get comprehensive patient summary"""
        if patient_id not in self.patient_data:
            return {"error": f"Patient {patient_id} not found"}
        
        patient_data = self.patient_data[patient_id]
        
        # Create summary
        summary = {
            "patient_id": patient_id,
            "basic_info": {
                "name": patient_data.get("name", "Unknown"),
                "age": patient_data.get("age", "Unknown"),
                "gender": patient_data.get("gender", "Unknown")
            },
            "medical_summary": {
                "conditions": patient_data.get("conditions", []),
                "medications": patient_data.get("medications", []),
                "allergies": patient_data.get("allergies", [])
            },
            "recent_activity": {
                "last_visit": patient_data.get("last_visit", "No recent visits"),
                "upcoming_appointments": patient_data.get("upcoming_appointments", [])
            },
            "interaction_history": len(patient_data.get("interactions", [])),
            "last_updated": patient_data.get("last_updated", "Never")
        }
        
        return summary
    
    def _rebuild_index(self):
        """Rebuild FAISS index from scratch"""
        try:
            # Create new index
            self.index = faiss.IndexFlatIP(self.dimension)
            
            if self.patient_data:
                texts = []
                ids = []
                for patient_id, data in self.patient_data.items():
                    text = self._create_searchable_text(data)
                    texts.append(text)
                    ids.append(patient_id)
                
                if texts:
                    embeddings = self.encoder.encode(texts)
                    self.index.add(embeddings.astype('float32'))
                    self.patient_embeddings = embeddings
                    self.patient_ids = ids
        except Exception as e:
            print(f"Error rebuilding index: {e}")
    
    def add_interaction(self, patient_id: str, interaction: Dict):
        """Add interaction to patient history"""
        if patient_id not in self.patient_data:
            self.patient_data[patient_id] = {"interactions": []}
        
        interaction["timestamp"] = datetime.now().isoformat()
        self.patient_data[patient_id]["interactions"].append(interaction)
        
        self._save_memory()
