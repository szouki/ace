"""
Embedding models for ACE framework.

This module provides embedders for semantic similarity and de-duplication.
"""

from ace.embeddings.base import BaseEmbedder
from ace.embeddings.openai_embedder import OpenAIEmbedder

__all__ = [
    "BaseEmbedder",
    "OpenAIEmbedder",
]

