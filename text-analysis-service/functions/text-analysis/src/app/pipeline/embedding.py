"""
Sentence embedding module using HuggingFace models.

Encapsulates HuggingFace logic for:
- Loading `all-MiniLM-L6-v2` model
- Implementing simple in-memory caching suitable for Lambda
- Exposing clean `embed(sentences: list[str])` interface

No clustering or sentiment logic here.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from functools import lru_cache

# TODO: Uncomment when transformers is available
# from transformers import AutoTokenizer, AutoModel
# import torch

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class EmbeddingModel:
    """
    Sentence embedding model wrapper with caching.
    
    Uses HuggingFace's all-MiniLM-L6-v2 model for sentence embeddings.
    Implements simple in-memory caching suitable for Lambda environments.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_size: int = 1000
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name
            cache_size: Maximum number of sentences to cache
        """
        self.model_name = model_name
        self.cache_size = cache_size
        
        # Initialize model lazily
        self._tokenizer = None
        self._model = None
        
        # Initialize cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.debug(f"EmbeddingModel initialized with model: {model_name}")
    
    def embed(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences: List of text sentences
            
        Returns:
            numpy array of shape (n_sentences, embedding_dim)
        """
        if not sentences:
            return np.array([])
        
        logger.debug(f"Generating embeddings for {len(sentences)} sentences")
        
        # Check cache for each sentence
        cached_embeddings = []
        uncached_sentences = []
        uncached_indices = []
        
        for i, sentence in enumerate(sentences):
            if sentence in self._embedding_cache:
                cached_embeddings.append(self._embedding_cache[sentence])
            else:
                uncached_sentences.append(sentence)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached sentences
        if uncached_sentences:
            logger.debug(f"Cache miss for {len(uncached_sentences)} sentences")
            new_embeddings = self._embed_uncached(uncached_sentences)
            
            # Update cache
            for sentence, embedding in zip(uncached_sentences, new_embeddings):
                self._update_cache(sentence, embedding)
        else:
            new_embeddings = np.array([])
        
        # Combine cached and new embeddings in original order
        if not cached_embeddings:
            # All sentences were uncached
            return new_embeddings
        elif not new_embeddings.size:
            # All sentences were cached
            return np.stack(cached_embeddings)
        else:
            # Mix of cached and uncached
            result = np.zeros((len(sentences), new_embeddings.shape[1]))
            
            # Place cached embeddings
            cached_idx = 0
            for i, sentence in enumerate(sentences):
                if sentence in self._embedding_cache:
                    result[i] = self._embedding_cache[sentence]
                else:
                    # Find position in uncached_sentences
                    uncached_pos = uncached_indices.index(i)
                    result[i] = new_embeddings[uncached_pos]
            
            return result
    
    def _embed_uncached(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for sentences not in cache.
        
        Args:
            sentences: List of text sentences not in cache
            
        Returns:
            numpy array of embeddings
        """
        # TODO: Implement actual HuggingFace model inference
        # For now, return placeholder embeddings for scaffolding
        
        logger.warning("Using placeholder embeddings - implement HuggingFace model")
        
        # Placeholder: random embeddings with deterministic seed
        np.random.seed(42)
        n_sentences = len(sentences)
        embedding_dim = 384  # all-MiniLM-L6-v2 has 384 dimensions
        
        # Generate deterministic "embeddings" based on sentence length
        embeddings = np.zeros((n_sentences, embedding_dim))
        for i, sentence in enumerate(sentences):
            # Simple deterministic hash for reproducibility
            seed = hash(sentence) % (2**32)
            np.random.seed(seed)
            embeddings[i] = np.random.randn(embedding_dim)
            
            # Normalize to unit length (cosine similarity expects normalized vectors)
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        
        return embeddings
    
    def _update_cache(self, sentence: str, embedding: np.ndarray) -> None:
        """
        Update the embedding cache.
        
        Args:
            sentence: Sentence text
            embedding: Corresponding embedding vector
        """
        if len(self._embedding_cache) >= self.cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
            logger.debug(f"Cache full, removed oldest entry: {oldest_key[:50]}...")
        
        self._embedding_cache[sentence] = embedding
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.debug(f"Cleared embedding cache ({cache_size} entries)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self._embedding_cache),
            'max_cache_size': self.cache_size,
            'cache_hit_rate': self._calculate_hit_rate() if hasattr(self, '_total_requests') else None
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (if tracking enabled)."""
        # TODO: Implement hit rate tracking if needed
        return 0.0


# Singleton instance for easy import
_default_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get or create the default embedding model instance.
    
    Returns:
        Shared EmbeddingModel instance
    """
    global _default_embedding_model
    if _default_embedding_model is None:
        _default_embedding_model = EmbeddingModel()
        logger.info("Created default embedding model instance")
    
    return _default_embedding_model


def embed_sentences(sentences: List[str]) -> np.ndarray:
    """
    Convenience function to embed sentences using the default model.
    
    Args:
        sentences: List of text sentences
        
    Returns:
        numpy array of embeddings
    """
    model = get_embedding_model()
    return model.embed(sentences)