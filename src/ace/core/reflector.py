"""
Reflector component for ACE framework.

The Reflector analyzes errors by comparing model outputs with ground truth,
identifies root causes, and tags bullets as helpful/harmful/neutral.
"""

from typing import Dict, Any, Optional, List
from ace.core.context import Context
from ace.core.schemas import REFLECTOR_SCHEMA
from ace.llm.base import BaseLLM


class Reflector:
    """
    Reflector component that analyzes errors and extracts insights.
    
    The Reflector:
    1. Takes model output and ground truth
    2. Identifies what went wrong and why
    3. Tags bullets as helpful/harmful/neutral
    4. Generates actionable insights for improvement
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: str
    ):
        """
        Initialize Reflector.
        
        Args:
            llm: LLM client to use for reflection
            prompt_template: Prompt template for reflection (required)
        """
        self.llm = llm
        self.prompt_template = prompt_template
    
    def reflect(
        self,
        context: Context,
        used_bullet_ids: List[str],
        prompt_variables: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze errors and generate reflection.
        
        Args:
            context: The context that was used (for bullet tagging)
            used_bullet_ids: List of bullet IDs that were used
            prompt_variables: Dictionary of variables to pass to the prompt template.
                             The used_bullets will automatically be formatted and added as 'used_bullets'.
            temperature: Optional temperature override
            
        Returns:
            Dictionary with:
                - reasoning: Analysis reasoning
                - error_identification: What went wrong
                - root_cause_analysis: Why it went wrong
                - correct_approach: What should have been done
                - key_insight: Key takeaway to remember
                - bullet_tags: List of {"id": bullet_id, "tag": "helpful/harmful/neutral"}
        """
        # Get the bullets that were used
        used_bullets = []
        for bullet_id in used_bullet_ids:
            bullet = context.get_bullet(bullet_id)
            if bullet:
                used_bullets.append({
                    "id": bullet_id,
                    "content": bullet.content,
                    "section": bullet.section,
                })
        
        # Prepare all variables for template
        all_variables = prompt_variables.copy() if prompt_variables else {}
        all_variables['used_bullets'] = "\n".join([
            f"- [{b['id']}] ({b['section']}): {b['content']}"
            for b in used_bullets
        ])
        
        # Build prompt
        prompt = self.prompt_template.format(**all_variables)
        
        # Generate reflection with structured output schema
        response = self.llm.generate_json(prompt, temperature=temperature, schema=REFLECTOR_SCHEMA)
        
        # Update context with bullet tags
        bullet_tags = response.get("bullet_tags", [])
        context.update_bullet_tags(bullet_tags)
        
        return response

