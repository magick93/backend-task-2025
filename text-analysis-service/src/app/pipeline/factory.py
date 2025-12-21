"""
Service factory for ML dependency injection.

Provides lazy-loaded singleton instances of embedding, clustering, and sentiment services
based on environment variable configuration.
"""

import os
from typing import Optional
from .interfaces import EmbeddingService, ClusteringService, SentimentService


class ServiceFactory:
    """
    Factory for creating ML service instances with lazy loading and singleton caching.
    
    Environment Variables:
        EMBEDDING_PROVIDER: 'local_hf' (default), 'openai' (future), 'bedrock' (future)
        CLUSTERING_PROVIDER: 'sklearn' (default)
        SENTIMENT_PROVIDER: 'vader' (default), 'transformers' (future)
    """
    
    # Singleton instances
    _embedding_service: Optional[EmbeddingService] = None
    _clustering_service: Optional[ClusteringService] = None
    _sentiment_service: Optional[SentimentService] = None
    
    @classmethod
    def get_embedding_service(cls) -> EmbeddingService:
        """
        Get or create the embedding service instance.
        
        Returns:
            EmbeddingService instance configured based on EMBEDDING_PROVIDER
        """
        if cls._embedding_service is None:
            provider = os.environ.get('EMBEDDING_PROVIDER', 'local_hf').lower()
            
            if provider == 'local_hf':
                from .embedding import HuggingFaceEmbeddingService
                cls._embedding_service = HuggingFaceEmbeddingService()
            elif provider == 'openai':
                # Future implementation
                raise NotImplementedError(f"Embedding provider '{provider}' not yet implemented")
            elif provider == 'bedrock':
                # Future implementation
                raise NotImplementedError(f"Embedding provider '{provider}' not yet implemented")
            else:
                raise ValueError(f"Unknown embedding provider: {provider}")
        
        return cls._embedding_service
    
    @classmethod
    def get_clustering_service(cls) -> ClusteringService:
        """
        Get or create the clustering service instance.
        
        Returns:
            ClusteringService instance configured based on CLUSTERING_PROVIDER
        """
        if cls._clustering_service is None:
            provider = os.environ.get('CLUSTERING_PROVIDER', 'sklearn').lower()
            
            if provider == 'sklearn':
                from .clustering import SklearnClusteringService
                cls._clustering_service = SklearnClusteringService()
            else:
                raise ValueError(f"Unknown clustering provider: {provider}")
        
        return cls._clustering_service
    
    @classmethod
    def get_sentiment_service(cls) -> SentimentService:
        """
        Get or create the sentiment service instance.
        
        Returns:
            SentimentService instance configured based on SENTIMENT_PROVIDER
        """
        if cls._sentiment_service is None:
            provider = os.environ.get('SENTIMENT_PROVIDER', 'vader').lower()
            
            if provider == 'vader':
                from .sentiment import VaderSentimentService
                cls._sentiment_service = VaderSentimentService()
            elif provider == 'transformers':
                # Future implementation
                raise NotImplementedError(f"Sentiment provider '{provider}' not yet implemented")
            else:
                raise ValueError(f"Unknown sentiment provider: {provider}")
        
        return cls._sentiment_service
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset all cached service instances (primarily for testing).
        """
        cls._embedding_service = None
        cls._clustering_service = None
        cls._sentiment_service = None