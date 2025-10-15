"""
Generator component for ACE framework.

The Generator uses the context to generate answers or code for tasks.
It tracks which bullets were helpful during generation.
"""

from typing import Dict, Any, Optional
from ace.core.context import Context
from ace.core.schemas import GENERATOR_SCHEMA
from ace.llm.base import BaseLLM


class Generator:
    """
    Generator component that uses context to generate task outputs.
    
    The Generator:
    1. Takes a task input and the current context
    2. Uses the context to inform its generation
    3. Tracks which bullet_ids were used/helpful
    4. Returns reasoning trace and final answer
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        context: Context,
        prompt_template: str
    ):
        """
        Initialize Generator.
        
        Args:
            llm: LLM client to use for generation
            context: Context to use for guidance
            prompt_template: Prompt template for generation (required)
        """
        self.llm = llm
        self.context = context
        self.prompt_template = prompt_template
    
    def generate(
        self,
        prompt_variables: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate output for a task using the context.
        
        Args:
            prompt_variables: Dictionary of variables to pass to the prompt template.
                             The context will automatically be formatted and added as 'context'.
            temperature: Optional temperature override
            
        Returns:
            Dictionary with:
                - reasoning: Chain of thought reasoning
                - bullet_ids: List of bullet IDs that were used
                - final_answer: The final answer/output (or other domain-specific output)
        """
        # Format context for prompt
        context_str = self.context.format_for_prompt(include_metadata=True)
        
        # Prepare all variables for template
        all_variables = prompt_variables.copy() if prompt_variables else {}
        all_variables['context'] = context_str
        
        # Build prompt
        prompt = self.prompt_template.format(**all_variables)
        
        # Generate JSON response with structured output schema
        response = self.llm.generate_json(prompt, temperature=temperature, schema=GENERATOR_SCHEMA)
        
        # Update bullet usage counts
        bullet_ids = response.get("bullet_ids", [])
        for bullet_id in bullet_ids:
            bullet = self.context.get_bullet(bullet_id)
            if bullet:
                bullet.usage_count += 1
        
        return response

