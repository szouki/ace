"""
OpenAI client implementation for ACE framework.
"""

import json
import os
from typing import Dict, Any, Optional, List
from openai import OpenAI

from ace.llm.base import BaseLLM


class OpenAIClient(BaseLLM):
    """
    OpenAI API client implementation.
    
    Supports GPT-5 and GPT-5-mini models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        organization: Optional[str] = None,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (e.g., "gpt-5", "gpt-5-mini")
            temperature: Default sampling temperature (GPT-5 only supports 1.0)
            max_tokens: Default max tokens to generate (None for no limit)
            organization: Optional OpenAI organization ID
        """
        # Get API key but don't store it as an instance variable for security
        _api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not _api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )
        
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        
        # Initialize OpenAI client (it stores the key internally in a secure way)
        self.client = OpenAI(
            api_key=_api_key,
            organization=organization,
        )
    
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
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            Generated text as a string
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from a prompt.
        
        Uses OpenAI's JSON mode (or structured outputs if schema provided) to ensure valid JSON output.
        
        Args:
            prompt: The input prompt (should request JSON output)
            temperature: Sampling temperature (overrides default, GPT-5 only supports 1.0)
            max_tokens: Maximum tokens to generate (overrides default)
            schema: Optional JSON schema for structured outputs. If provided, OpenAI will enforce this schema.
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            Parsed JSON as a dictionary
            
        Raises:
            ValueError: If the output is not valid JSON
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare API parameters
        api_params = {
            "model": self.model,
            "messages": messages,
        }
        
        # Only add max_completion_tokens if specified
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        if max_tok is not None:
            api_params["max_completion_tokens"] = max_tok
        
        # Use structured outputs if schema is provided, otherwise use JSON mode
        if schema:
            api_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": schema
                }
            }
        else:
            api_params["response_format"] = {"type": "json_object"}
        
        # Only include temperature if it's 1.0 (GPT-5 only supports default temperature)
        temp = temperature if temperature is not None else self.default_temperature
        if temp == 1.0:
            api_params["temperature"] = temp
        
        # Call OpenAI API
        response = self.client.chat.completions.create(**api_params, **kwargs)
        
        content = response.choices[0].message.content
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content}")
    
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
            temperature: Sampling temperature (overrides default, GPT-5 only supports 1.0)
            max_tokens: Maximum tokens to generate (overrides default)
            **kwargs: Additional OpenAI API parameters
            
        Returns:
            Generated text as a string
        """
        # Prepare API parameters
        api_params = {
            "model": self.model,
            "messages": messages,
        }
        
        # Only add max_completion_tokens if specified
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        if max_tok is not None:
            api_params["max_completion_tokens"] = max_tok
        
        # Only include temperature if it's 1.0 (GPT-5 only supports default temperature)
        temp = temperature if temperature is not None else self.default_temperature
        if temp == 1.0:
            api_params["temperature"] = temp
        
        response = self.client.chat.completions.create(**api_params, **kwargs)
        
        return response.choices[0].message.content
    
    def __repr__(self) -> str:
        return f"OpenAIClient(model={self.model})"

