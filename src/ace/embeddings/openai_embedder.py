"""
OpenAI embedder implementation for ACE framework.
"""

import os
import numpy as np
from typing import List, Union, Optional
from openai import OpenAI

from ace.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embeddings API client implementation.
    
    Uses OpenAI's text-embedding models (text-embedding-3-small, text-embedding-3-large).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        organization: Optional[str] = None,
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model name (e.g., "text-embedding-3-small", "text-embedding-3-large")
            dimensions: Optional dimension reduction (must be <= model's default dimension)
            organization: Optional OpenAI organization ID
        """
        # Get API key but don't store it as an instance variable for security
        _api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not _api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )
        
        self.model = model
        self.dimensions = dimensions
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=_api_key,
            organization=organization,
        )
        
        # Set default dimensions based on model
        if model == "text-embedding-3-small":
            self._default_dim = 1536
        elif model == "text-embedding-3-large":
            self._default_dim = 3072
        elif model == "text-embedding-ada-002":
            self._default_dim = 1536
        else:
            self._default_dim = 1536  # Default fallback
        
        # Validate custom dimensions
        if dimensions is not None:
            if dimensions > self._default_dim:
                raise ValueError(
                    f"Requested dimensions ({dimensions}) exceeds model's default ({self._default_dim})"
                )
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.dimensions if self.dimensions is not None else self._default_dim
    
    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text string(s).
        
        Args:
            text: A single text string or list of text strings
            
        Returns:
            If input is a string: numpy array of shape (embedding_dim,)
            If input is a list: list of numpy arrays, each of shape (embedding_dim,)
        """
        # Handle single string vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Clean text (replace newlines with spaces as recommended by OpenAI)
        texts = [t.replace("\n", " ") for t in texts]
        
        # Prepare API parameters
        api_params = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        
        # Add dimensions parameter if specified
        if self.dimensions is not None:
            api_params["dimensions"] = self.dimensions
        
        # Call OpenAI API
        response = self.client.embeddings.create(**api_params)
        
        # Extract embeddings and convert to numpy arrays
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        
        # Return single array or list based on input
        return embeddings[0] if is_single else embeddings
    
    def __repr__(self) -> str:
        dim_str = f", dimensions={self.dimensions}" if self.dimensions else ""
        return f"OpenAIEmbedder(model={self.model}{dim_str})"

