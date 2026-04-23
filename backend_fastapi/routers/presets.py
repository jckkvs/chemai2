"""backend_fastapi/routers/presets.py - プリセット/履歴管理API"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json, os, logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/presets", tags=["presets"])

PRESET_DIR = os.getenv("PRESET_DIR", "./data/presets")
os.makedirs(PRESET_DIR, exist_ok=True)

class PresetSave(BaseModel):
    name: str
    description: str = ""
    config: Dict[str, Any]

class HistoryEntry(BaseModel):
    id: str
    timestamp: str
    filename: str
    best_model: str
    score: float

@router.post("/save")
async def save_preset(req: PresetSave):
    try:
        path = os.path.join(PRESET_DIR, f"{req.name.replace(' ','_')}.json")
        if os.path.exists(path): 
            raise HTTPException(400, "プリセット名が重複しています")
            
        preset_data = {
            "name": req.name, 
            "desc": req.description, 
            "config": req.config, 
            "created_at": datetime.utcnow().isoformat()
        }
        
        with open(path, "w", encoding="utf-8") as f: 
            json.dump(preset_data, f, ensure_ascii=False, indent=2)
            
        return {"status": "saved", "path": path}
    except Exception as e:
        logger.error(f"Failed to save preset: {e}")
        raise HTTPException(500, str(e))

@router.get("/list")
async def list_presets():
    try:
        presets = []
        for f in os.listdir(PRESET_DIR):
            if f.endswith(".json") and f != "history.json": # Exclude special files if any
                try:
                    with open(os.path.join(PRESET_DIR, f), "r", encoding="utf-8") as fh:
                        presets.append(json.load(fh))
                except Exception:
                    continue
        return presets
    except Exception as e:
        logger.error(f"Failed to list presets: {e}")
        return []

@router.delete("/{name}")
async def delete_preset(name: str):
    try:
        path = os.path.join(PRESET_DIR, f"{name.replace(' ','_')}.json")
        if os.path.exists(path): 
            os.remove(path)
            return {"status": "deleted"}
        raise HTTPException(404, "見つかりません")
    except Exception as e:
        logger.error(f"Failed to delete preset: {e}")
        raise HTTPException(500, str(e))

@router.post("/history/record")
async def record_history(entry: HistoryEntry):
    try:
        hist_path = os.path.join(PRESET_DIR, "history.jsonl")
        with open(hist_path, "a", encoding="utf-8") as f: 
            f.write(json.dumps(entry.model_dump()) + "\n")
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record history: {e}")
        raise HTTPException(500, str(e))

@router.get("/history")
async def get_history(limit: int = 20):
    try:
        hist_path = os.path.join(PRESET_DIR, "history.jsonl")
        if not os.path.exists(hist_path): 
            return []
        with open(hist_path, "r", encoding="utf-8") as f: 
            lines = f.readlines()
        return [json.loads(l) for l in lines[-limit:]]
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []
