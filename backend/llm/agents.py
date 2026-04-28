# backend/llm/agents.py

import json
import logging
from typing import Dict, Any, List, Optional
from backend.llm.engine import LLMEngine
from backend.llm.prompts import (
    SYSTEM_PROMPT_INTENT,
    SYSTEM_PROMPT_WORKFLOW,
    SYSTEM_PROMPT_INTERPRETATION,
    SYSTEM_PROMPT_REPORT
)

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, engine: LLMEngine):
        self.engine = engine

class IntentAgent(BaseAgent):
    async def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract task, domain, target, and constraints from natural language"""
        prompt = f"User Request: {query}"
        # For simplicity, we use the engine's non-streaming call with a system prompt override logic
        # Note: Current LLMEngine doesn't support system prompt override easily, 
        # so we'll prepend it for now.
        full_prompt = f"{SYSTEM_PROMPT_INTENT}\n\n{prompt}\n\nReturn JSON only."
        
        try:
            # We use a high-level call if available, or just use the model directly
            # For this implementation, we assume engine.generate_report can take a custom prompt
            response = await self.engine.generate_report({"query": full_prompt}, template="custom")
            # Parse JSON from response
            # Sometimes LLMs wrap JSON in code blocks
            clean_res = response.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean_res)
        except Exception as e:
            logger.error(f"Intent extraction failed: {str(e)}")
            return {"task": "unknown", "domain": "unknown", "target": "unknown", "constraints": []}

class WorkflowAgent(BaseAgent):
    async def select_workflow(self, intent: Dict[str, Any]) -> List[Dict[str, str]]:
        """Select the best MI workflow based on intent"""
        prompt = f"{SYSTEM_PROMPT_WORKFLOW}\n\nUser Intent: {json.dumps(intent)}\n\nReturn a JSON list of steps."
        try:
            response = await self.engine.generate_report({"query": prompt}, template="custom")
            clean_res = response.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean_res)
        except Exception as e:
            logger.error(f"Workflow selection failed: {str(e)}")
            return [{"step": "data_validation", "reason": "Default safety step"}]

class InterpretationAgent(BaseAgent):
    async def interpret_results(self, analysis_results: Dict[str, Any]) -> str:
        """Translate ML results into chemical insights"""
        prompt = f"{SYSTEM_PROMPT_INTERPRETATION}\n\nAnalysis Results: {json.dumps(analysis_results)}"
        return await self.engine.generate_report({"query": prompt}, template="custom")

class MIAnalysisAgent:
    """Orchestrator for the autonomous MI analysis platform"""
    
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.intent_agent = IntentAgent(engine)
        self.workflow_agent = WorkflowAgent(engine)
        self.interpretation_agent = InterpretationAgent(engine)
        # Parameter and Report agents to be added in future iterations
        
    async def process_request(self, user_query: str):
        """Main entry point for autonomous analysis"""
        logger.info(f"Processing MI request: {user_query}")
        
        # 1. Understand Intent
        intent = await self.intent_agent.extract_intent(user_query)
        logger.info(f"Extracted Intent: {intent}")
        
        # 2. Select Workflow
        workflow = await self.workflow_agent.select_workflow(intent)
        logger.info(f"Selected Workflow: {workflow}")
        
        # 3. Parameter Optimization (Placeholder)
        # parameters = await self.param_agent.suggest(workflow)
        
        # 4. Return the plan to the user for confirmation
        return {
            "intent": intent,
            "workflow": workflow,
            "status": "ready_for_execution"
        }

    async def finalize_analysis(self, execution_results: Dict[str, Any]):
        """Interpret results and generate report after execution"""
        interpretation = await self.interpretation_agent.interpret_results(execution_results)
        
        # In a real scenario, this would then call the ReportAgent
        return {
            "interpretation": interpretation,
            "status": "completed"
        }
