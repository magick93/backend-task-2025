"""
Sentence embedding module using HuggingFace models.

Encapsulates HuggingFace logic for:
- Loading `all-MiniLM-L6-v2` model
- Implementing simple in-memory caching suitable for Lambda
- Exposing clean `embed(sentences: list[str])` interface

No clustering or sentiment logic here.
"""

import os
from typing import List, Optional, Dict, Any
import numpy as np
from functools import lru_cache

# Set HF_HOME to /tmp/huggingface for Lambda compatibility
# This must be done before importing transformers/sentence_transformers
os.environ['HF_HOME'] = '/tmp/huggingface'

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Fallback for when dependencies aren't installed (e.g. during some local tests)
    SentenceTransformer = None

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
        cache_size: int = 1000,
        device: Optional[str] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name
            cache_size: Maximum number of sentences to cache
            device: Device to run model on ('cpu', 'cuda', etc.). If None, auto-detects.
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.device = device
        
        # Initialize model lazily
        self._model = None
        
        # Initialize cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.debug(f"EmbeddingModel initialized with model: {model_name}, device: {device}")
    
    @property
    def model(self):
        """Lazy loading of the SentenceTransformer model."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed")
            
            logger.info(f"Loading embedding model: {self.model_name}...")
            # cache_folder is automatically handled by HF_HOME environment variable
            self._model = SentenceTransformer(self.model_name)
            
            # Move model to specified device if provided
            if self.device is not None:
                logger.debug(f"Moving model to device: {self.device}")
                self._model = self._model.to(self.device)
            else:
                # Auto-detect: try CUDA, fallback to CPU
                try:
                    import torch
                    # Force CPU in test environment or if CUDA is incompatible
                    if os.environ.get('ENVIRONMENT') == 'test':
                        logger.debug("Test environment detected, forcing CPU")
                        self._model = self._model.to('cpu')
                    elif torch.cuda.is_available():
                        logger.debug("CUDA available, attempting to move model to CUDA")
                        try:
                            self._model = self._model.to('cuda')
                        except (RuntimeError, torch.cuda.CudaError) as e:
                            logger.warning(f"CUDA error moving model to GPU: {e}. Falling back to CPU.")
                            self._model = self._model.to('cpu')
                    else:
                        logger.debug("CUDA not available, using CPU")
                except ImportError:
                    logger.debug("Torch not available, using default device")
            
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
            
        return self._model
    
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
        # Use the actual model for encoding
        embeddings = self.model.encode(sentences)
        
        # Ensure it returns numpy array (SentenceTransformer usually does, but to be safe)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
            
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
