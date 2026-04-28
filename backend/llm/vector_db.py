# backend/llm/vector_db.py

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

logger = logging.getLogger(__name__)

class MIKnowledgeBase:
    """Vector database for storing and retrieving MI protocols and domain knowledge"""
    
    def __init__(self, db_path: str = "data/chroma_db"):
        self.db_path = db_path
        self.client = None
        self.collection = None
        
        if not HAS_CHROMA:
            logger.warning("ChromaDB not installed. RAG features will be disabled.")
            return

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="mi_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB initialized at {db_path}")

    def add_protocol(self, id: str, content: str, metadata: Dict[str, Any]):
        """Add a MI analysis protocol to the vector store"""
        if not self.collection:
            return
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[id]
        )
        logger.info(f"Added protocol: {id}")

    def query_knowledge(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant protocols or knowledge snippets"""
        if not self.collection:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        output = []
        for i in range(len(results['documents'][0])):
            output.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        return output

# Singleton instance for the app
knowledge_base = MIKnowledgeBase()
