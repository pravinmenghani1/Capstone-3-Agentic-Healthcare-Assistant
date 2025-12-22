"""
Medical Information Search Tools with RAG capabilities
"""

import json
import os
import requests
from typing import Dict, List, Any
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SearchTools:
    def __init__(self, data_dir: str = "src/data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize sentence transformer with error handling
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: SentenceTransformer failed ({e}), using fallback")
            self.encoder = None
        
        # Medical knowledge base (mock data)
        self.knowledge_base = self._load_medical_knowledge()
        
        # Initialize FAISS index for RAG
        self.rag_index = None
        self.rag_documents = []
        if self.encoder is not None:
            self._build_rag_index()
    
    def _load_medical_knowledge(self) -> List[Dict]:
        """Load medical knowledge base"""
        # In production, this would load from medical databases
        return [
            {
                "id": "ckd_001",
                "title": "Chronic Kidney Disease Management",
                "content": "Chronic kidney disease (CKD) is a progressive loss of kidney function. Management includes blood pressure control, diabetes management, dietary modifications, and regular monitoring of kidney function. ACE inhibitors and ARBs are first-line treatments for slowing progression.",
                "source": "National Kidney Foundation",
                "category": "nephrology",
                "last_updated": "2024-01-15"
            },
            {
                "id": "ckd_002", 
                "title": "Latest CKD Treatment Approaches",
                "content": "Recent advances in CKD treatment include SGLT2 inhibitors, which have shown nephroprotective effects. Finerenone, a non-steroidal MRA, has demonstrated cardiovascular and kidney benefits in diabetic CKD patients. Early referral to nephrology is crucial for optimal outcomes.",
                "source": "American Journal of Nephrology",
                "category": "nephrology",
                "last_updated": "2024-02-20"
            },
            {
                "id": "htn_001",
                "title": "Hypertension Management Guidelines",
                "content": "Current hypertension guidelines recommend lifestyle modifications as first-line therapy, including dietary changes, exercise, and weight management. Pharmacological treatment typically starts with ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics.",
                "source": "American Heart Association",
                "category": "cardiology", 
                "last_updated": "2024-01-10"
            },
            {
                "id": "dm_001",
                "title": "Type 2 Diabetes Treatment Updates",
                "content": "Modern diabetes management emphasizes individualized HbA1c targets, typically <7% for most adults. Metformin remains first-line therapy. GLP-1 agonists and SGLT2 inhibitors offer cardiovascular and renal benefits beyond glucose control.",
                "source": "American Diabetes Association",
                "category": "endocrinology",
                "last_updated": "2024-02-05"
            },
            {
                "id": "migraine_001",
                "title": "Migraine Prevention and Treatment",
                "content": "Migraine treatment includes acute therapies (triptans, NSAIDs) and preventive medications (beta-blockers, anticonvulsants, CGRP inhibitors). Lifestyle modifications including sleep hygiene, stress management, and trigger avoidance are essential components.",
                "source": "American Headache Society",
                "category": "neurology",
                "last_updated": "2024-01-25"
            }
        ]
    
    def _build_rag_index(self):
        """Build FAISS index for RAG retrieval"""
        try:
            if not self.knowledge_base or self.encoder is None:
                return
            
            # Create document texts for embedding
            documents = []
            for item in self.knowledge_base:
                doc_text = f"{item['title']} {item['content']}"
                documents.append(doc_text)
                self.rag_documents.append(item)
            
            # Generate embeddings
            embeddings = self.encoder.encode(documents)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.rag_index = faiss.IndexFlatIP(dimension)
            self.rag_index.add(embeddings.astype('float32'))
            
        except Exception as e:
            print(f"Error building RAG index: {e}")
            self.rag_index = None
    
    def search_medical_info(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search medical information using RAG"""
        try:
            if not self.rag_index or not self.rag_documents or self.encoder is None:
                return self._fallback_search(query)
            
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Search similar documents
            scores, indices = self.rag_index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score > 0.1:  # Similarity threshold
                    doc = self.rag_documents[idx].copy()
                    doc['relevance_score'] = float(score)
                    doc['rank'] = i + 1
                    results.append(doc)
            
            # If no good matches, try web search
            if not results:
                results = self._web_search(query)
            
            return results
            
        except Exception as e:
            print(f"Error in medical search: {e}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> List[Dict]:
        """Fallback search using keyword matching"""
        results = []
        query_lower = query.lower()
        
        for doc in self.knowledge_base:
            # Simple keyword matching
            title_match = any(word in doc['title'].lower() for word in query_lower.split())
            content_match = any(word in doc['content'].lower() for word in query_lower.split())
            
            if title_match or content_match:
                doc_copy = doc.copy()
                doc_copy['relevance_score'] = 0.8 if title_match else 0.6
                results.append(doc_copy)
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:5]
    
    def _web_search(self, query: str) -> List[Dict]:
        """Mock web search for medical information"""
        # In production, integrate with medical APIs like PubMed, WHO, etc.
        mock_results = [
            {
                "id": f"web_{hash(query) % 1000}",
                "title": f"Latest Research on {query.title()}",
                "content": f"Recent studies on {query} show promising results in treatment approaches. Clinical trials demonstrate improved patient outcomes with new therapeutic interventions.",
                "source": "PubMed Central",
                "category": "research",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "relevance_score": 0.7,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/search?q={query.replace(' ', '+')}"
            },
            {
                "id": f"web_{hash(query + 'who') % 1000}",
                "title": f"WHO Guidelines for {query.title()}",
                "content": f"World Health Organization provides comprehensive guidelines for {query} management, including prevention strategies and treatment protocols.",
                "source": "World Health Organization",
                "category": "guidelines",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "relevance_score": 0.8,
                "url": f"https://www.who.int/search?q={query.replace(' ', '+')}"
            }
        ]
        
        return mock_results
    
    def get_disease_info(self, disease_name: str) -> Dict:
        """Get comprehensive disease information"""
        try:
            # Search for disease-specific information
            search_results = self.search_medical_info(disease_name)
            
            if not search_results:
                return {
                    "disease": disease_name,
                    "status": "not_found",
                    "message": f"No information found for {disease_name}"
                }
            
            # Compile comprehensive info
            disease_info = {
                "disease": disease_name,
                "status": "found",
                "overview": search_results[0]['content'] if search_results else "",
                "treatment_options": [],
                "latest_research": [],
                "guidelines": [],
                "sources": []
            }
            
            # Categorize results
            for result in search_results:
                category = result.get('category', 'general')
                source = result.get('source', 'Unknown')
                
                if 'treatment' in result['title'].lower() or 'therapy' in result['content'].lower():
                    disease_info['treatment_options'].append({
                        "title": result['title'],
                        "description": result['content'][:200] + "...",
                        "source": source
                    })
                elif 'research' in result['title'].lower() or 'study' in result['content'].lower():
                    disease_info['latest_research'].append({
                        "title": result['title'],
                        "description": result['content'][:200] + "...",
                        "source": source
                    })
                elif 'guideline' in result['title'].lower():
                    disease_info['guidelines'].append({
                        "title": result['title'],
                        "description": result['content'][:200] + "...",
                        "source": source
                    })
                
                if source not in disease_info['sources']:
                    disease_info['sources'].append(source)
            
            return disease_info
            
        except Exception as e:
            return {
                "disease": disease_name,
                "status": "error",
                "message": f"Error retrieving disease information: {str(e)}"
            }
    
    def get_drug_interactions(self, medications: List[str]) -> Dict:
        """Check for drug interactions (mock implementation)"""
        # In production, integrate with drug interaction databases
        interactions = []
        
        # Mock interaction data
        known_interactions = {
            ("metformin", "contrast"): "Increased risk of lactic acidosis",
            ("warfarin", "aspirin"): "Increased bleeding risk",
            ("ace_inhibitor", "potassium"): "Risk of hyperkalemia"
        }
        
        # Check for interactions
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                key = tuple(sorted([med1.lower(), med2.lower()]))
                if key in known_interactions:
                    interactions.append({
                        "medications": [med1, med2],
                        "interaction": known_interactions[key],
                        "severity": "moderate"
                    })
        
        return {
            "medications": medications,
            "interactions_found": len(interactions),
            "interactions": interactions,
            "status": "checked"
        }
    
    def get_treatment_guidelines(self, condition: str) -> Dict:
        """Get treatment guidelines for a condition"""
        guidelines_results = self.search_medical_info(f"{condition} treatment guidelines")
        
        return {
            "condition": condition,
            "guidelines": guidelines_results,
            "last_updated": datetime.now().isoformat()
        }
