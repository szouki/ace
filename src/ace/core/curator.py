"""
Curator component for ACE framework.

The Curator manages context evolution by identifying new insights from reflections
and adding them to the context through incremental ADD operations.
"""

from typing import Dict, Any, Optional
from ace.core.context import Context
from ace.core.schemas import CURATOR_SCHEMA
from ace.llm.base import BaseLLM


class Curator:
    """
    Curator component that manages context evolution.
    
    The Curator:
    1. Reviews reflections and current context
    2. Identifies NEW insights missing from context
    3. Generates ADD operations only (incremental updates)
    4. Avoids redundancy with existing content
    5. Implements grow-and-refine principle
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: str
    ):
        """
        Initialize Curator.
        
        Args:
            llm: LLM client to use for curation
            prompt_template: Prompt template for curation (required)
        """
        self.llm = llm
        self.prompt_template = prompt_template
    
    def curate(
        self,
        context: Context,
        prompt_variables: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Curate new insights to add to the context.
        
        Args:
            context: Current context (will be updated with new insights)
            prompt_variables: Dictionary of variables to pass to the prompt template.
                             The current_context and context_stats will automatically be added.
            temperature: Optional temperature override
            
        Returns:
            Dictionary with:
                - reasoning: Curation reasoning
                - operations: List of operations (ADD, UPDATE, DELETE)
                  Each operation has: {"type": str, "section": str (for ADD), "content": str, "bullet_id": str (for UPDATE/DELETE)}
        """
        # Get context stats
        context_stats = context.get_stats()
        
        # Format current context
        context_str = context.format_for_prompt(include_metadata=True)
        
        # Prepare all variables for template
        all_variables = prompt_variables.copy() if prompt_variables else {}
        all_variables['current_context'] = context_str
        all_variables['context_stats'] = context_stats
        
        # Build prompt
        prompt = self.prompt_template.format(**all_variables)
        
        # Generate curation response with structured output schema
        response = self.llm.generate_json(prompt, temperature=temperature, schema=CURATOR_SCHEMA)
        
        # Apply operations to context
        operations = response.get("operations", [])
        for operation in operations:
            op_type = operation.get("type")
            if op_type == "ADD":
                section = operation.get("section")
                content = operation.get("content")
                # Skip if section or content is None/null
                if section and content:
                    try:
                        context.add_bullet(section=section, content=content)
                    except ValueError as e:
                        # Section doesn't exist or max bullets reached
                        print(f"Warning: Could not add bullet to section '{section}': {e}")
        
        return response

