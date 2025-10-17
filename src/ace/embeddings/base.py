"""
Base embedder interface for ACE framework.

This abstract base class defines the interface that all embedder implementations must follow,
enabling easy swapping between different providers (OpenAI, Transformers, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding models.
    
    All embedder implementations must inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text string(s).
        
        Args:
            text: A single text string or list of text strings
            
        Returns:
            If input is a string: numpy array of shape (embedding_dim,)
            If input is a list: list of numpy arrays, each of shape (embedding_dim,)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            Integer dimension of embedding vectors
        """
        pass

