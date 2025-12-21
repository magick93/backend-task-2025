"""
Abstract base classes (interfaces) for ML services in the text analysis pipeline.

Defines the contracts for embedding, clustering, and sentiment analysis services,
enabling dependency injection and easy swapping of implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class EmbeddingService(ABC):
    """Interface for sentence embedding services."""
    
    @abstractmethod
    def embed(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences: List of text sentences
            
        Returns:
            numpy array of shape (n_sentences, embedding_dim)
        """
        pass


class ClusteringService(ABC):
    """Interface for clustering services."""
    
    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings and return labels (-1 for noise).
        
        Args:
            embeddings: numpy array of shape (n_sentences, embedding_dim)
            
        Returns:
            Array of cluster labels (-1 for noise)
        """
        pass


class SentimentService(ABC):
    """Interface for sentiment analysis services."""
    
    @abstractmethod
    def analyze_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of sentences.
        
        Args:
            sentences: List of text sentences
            
        Returns:
            List of sentiment analysis results, each containing at least:
                - 'label': str ('positive', 'negative', 'neutral')
                - 'confidence': float
                - 'scores': Dict[str, float] (optional)
                - 'method': str (optional)
        """
        pass
    
    def get_cluster_sentiment(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        precomputed_sentiments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculate aggregate sentiment for clusters.
        
        This default implementation uses analyze_batch to compute sentiment
        for each sentence, then aggregates by cluster. Subclasses may override
        for more efficient implementations.
        
        Args:
            sentences: List of all sentences
            embeddings: Sentence embeddings (unused in default implementation)
            labels: Cluster labels (-1 for noise)
            precomputed_sentiments: Optional list of pre-computed sentiment results
                                    matching sentences order
            
        Returns:
            Dictionary mapping cluster label to sentiment analysis containing:
                - 'sentiment': str overall sentiment label
                - 'confidence': float aggregate confidence
                - 'strength': float proportion of dominant sentiment
                - 'proportions': Dict[str, float] distribution of sentiment labels
                - 'counts': Dict[str, int] raw counts
                - 'total_sentences': int
                - 'avg_compound': float average compound score (if available)
        """
        from typing import Counter
        import numpy as np
        
        cluster_sentiments = {}
        unique_labels = np.unique(labels)
        
        # Analyze all sentences once if not precomputed
        if precomputed_sentiments is None:
            sentiment_results = self.analyze_batch(sentences)
        else:
            sentiment_results = precomputed_sentiments
        
        if len(sentiment_results) != len(sentences):
            raise ValueError(
                f"Mismatch between sentences ({len(sentences)}) and "
                f"sentiment results ({len(sentiment_results)})"
            )
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            
            cluster_indices = np.where(labels == label)[0]
            cluster_results = [sentiment_results[i] for i in cluster_indices]
            
            if not cluster_results:
                continue
            
            # Count sentiment labels
            label_counts = Counter(result['label'] for result in cluster_results)
            total = len(cluster_results)
            
            # Calculate proportions
            proportions = {
                label: count / total 
                for label, count in label_counts.items()
            }
            
            # Determine overall sentiment (majority vote)
            overall_sentiment = max(label_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate average confidence
            avg_confidence = np.mean([
                result.get('confidence', 0.0) for result in cluster_results
            ])
            
            # Calculate sentiment strength (proportion of dominant sentiment)
            sentiment_strength = proportions.get(overall_sentiment, 0)
            
            # Try to compute average compound score if available
            compound_scores = []
            for result in cluster_results:
                scores = result.get('scores', {})
                if 'compound' in scores:
                    compound_scores.append(scores['compound'])
            avg_compound = np.mean(compound_scores) if compound_scores else 0.0
            
            cluster_sentiments[label] = {
                'sentiment': overall_sentiment,
                'confidence': float(avg_confidence),
                'strength': float(sentiment_strength),
                'proportions': proportions,
                'counts': dict(label_counts),
                'total_sentences': total,
                'avg_compound': float(avg_compound)
            }
        
        return cluster_sentiments