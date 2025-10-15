"""
Prompt management utilities for ACE framework.
"""

import os
import yaml
from typing import Dict, Optional, Any


class PromptManager:
    """
    Manages prompt templates for different domains.
    
    Loads prompts from YAML configuration files.
    """
    
    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize PromptManager.
        
        Args:
            prompts_file: Path to the prompts YAML file (optional)
        """
        self.prompts_file = prompts_file
        self._prompts_cache: Dict[str, Dict[str, str]] = {}
        self._current_prompts: Dict[str, str] = {}
    
    def load_prompts(self, prompts_file: Optional[str] = None) -> Dict[str, str]:
        """
        Load prompts from YAML file.
        
        Args:
            prompts_file: Path to prompts YAML file (overrides the one from __init__)
            
        Returns:
            Dictionary with component names as keys and prompt text as values
        """
        # Use provided file or fall back to the one from init
        file_to_load = prompts_file or self.prompts_file
        
        if not file_to_load:
            raise ValueError("No prompts file specified")
        
        # Return cached prompts if already loaded for this file
        if file_to_load in self._prompts_cache:
            self._current_prompts = self._prompts_cache[file_to_load]
            return self._current_prompts
        
        if not os.path.exists(file_to_load):
            raise FileNotFoundError(f"Prompts file not found: {file_to_load}")
        
        with open(file_to_load, "r") as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Invalid prompts file format: {file_to_load}")
        
        # Extract prompts (expected keys: generator, reflector, curator)
        prompts = {
            key: value
            for key, value in data.items()
            if isinstance(value, str)
        }
        
        self._prompts_cache[file_to_load] = prompts
        self._current_prompts = prompts
        
        return prompts
    
    def get_prompt(self, component: str, prompts_file: Optional[str] = None) -> Optional[str]:
        """
        Get a specific prompt for a component.
        
        Args:
            component: Component name (e.g., "generator", "reflector", "curator")
            prompts_file: Path to prompts YAML file (optional, overrides the one from __init__)
            
        Returns:
            Prompt text or None if not found
        """
        prompts = self.load_prompts(prompts_file)
        return prompts.get(component)

