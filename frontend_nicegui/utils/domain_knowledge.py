"""
Domain Knowledge Manager for ChemAI2
Handles casual domain knowledge input and storage
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class DomainKnowledgeManager:
    """Simple manager for user's casual domain knowledge"""

    def __init__(self, storage_path: str = None):
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(__file__).parent.parent.parent / "data" / "domain_knowledge.json"
        self.knowledge_items: List[Dict[str, Any]] = []
        self.load()

    def add_knowledge(
        self,
        knowledge_type: str,
        content: str,
        context: str = "",
        page: str = "",
    ) -> Dict[str, Any]:
        """
        Add a piece of casual domain knowledge.

        Args:
            knowledge_type: Type of knowledge ('variable_property', 'constraint', 'system', 'other')
            content: The actual knowledge content (e.g., "Temperature goes up, refractive index goes down")
            context: Additional context (e.g., "Refractive index prediction")
            page: Which page this was input from

        Returns:
            The created knowledge item
        """
        item = {
            "id": len(self.knowledge_items) + 1,
            "type": knowledge_type,
            "content": content,
            "context": context,
            "page": page,
            "timestamp": datetime.now().isoformat(),
            "used": False,
            "structured": None  # Will be filled by LLM later
        }
        self.knowledge_items.append(item)
        self.save()
        return item

    def get_by_type(self, knowledge_type: str) -> List[Dict[str, Any]]:
        """Get all knowledge items of a specific type"""
        return [item for item in self.knowledge_items if item["type"] == knowledge_type]

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all knowledge items"""
        return self.knowledge_items

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        """Search knowledge items by keyword"""
        keyword_lower = keyword.lower()
        return [
            item
            for item in self.knowledge_items
            if keyword_lower in item["content"].lower()
            or keyword_lower in item["context"].lower()
        ]

    def mark_used(self, item_id: int) -> None:
        """Mark a knowledge item as used"""
        for item in self.knowledge_items:
            if item["id"] == item_id:
                item["used"] = True
                break
        self.save()

    def save(self) -> None:
        """Save knowledge items to JSON file"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "1.0",
                    "updated_at": datetime.now().isoformat(),
                    "items": self.knowledge_items,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self) -> None:
        """Load knowledge items from JSON file"""
        if not self.storage_path.exists():
            self.knowledge_items = []
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.knowledge_items = data.get("items", [])
        except Exception:
            self.knowledge_items = []

    def clear(self) -> None:
        """Clear all knowledge items"""
        self.knowledge_items = []
        self.save()

    def get_summary(self) -> str:
        """Get a summary of all knowledge for LLM context"""
        if not self.knowledge_items:
            return "No domain knowledge saved yet."

        summary = "=== User's Domain Knowledge ===\n"
        for item in self.knowledge_items:
            summary += f"[{item['type']}] {item['content']}\n"
            if item['context']:
                summary += f"  Context: {item['context']}\n"
        return summary


# Global instance
domain_knowledge = DomainKnowledgeManager()
