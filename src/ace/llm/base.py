"""
Base LLM interface for ACE framework.

This abstract base class defines the interface that all LLM clients must implement,
enabling easy swapping between different providers (OpenAI, Anthropic, local models, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseLLM(ABC):
    """
    Abstract base class for LLM clients.
    
    All LLM implementations must inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion from a prompt.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text as a string
        """
        pass
    
    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from a prompt.
        
        This method should ensure the output is valid JSON and parse it
        into a Python dictionary.
        
        Args:
            prompt: The input prompt (should request JSON output)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Parsed JSON as a dictionary
            
        Raises:
            ValueError: If the output is not valid JSON
        """
        pass
    
    @abstractmethod
    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a list of messages (chat format).
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     e.g., [{"role": "user", "content": "Hello"}]
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text as a string
        """
        pass

