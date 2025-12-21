"""
Clustering module for sentence embeddings.

Uses density-based clustering (DBSCAN) to group similar sentences.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import DBSCAN

from ..utils.logging import setup_logger
from .interfaces import ClusteringService

logger = setup_logger(__name__)


class SklearnClusteringService(ClusteringService):
    """
    Scikit-learn based clustering service using DBSCAN.
    """
    
    def __init__(
        self,
        eps: float = 0.3,  # Corresponds to similarity threshold of ~0.7 with cosine distance
        min_samples: int = 2,
        metric: str = 'cosine'
    ):
        """
        Initialize the cluster analyzer.
        
        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            metric: The metric to use when calculating distance between instances in a feature array.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        logger.debug(
            f"SklearnClusteringService initialized: eps={eps}, "
            f"min_samples={min_samples}, metric={metric}"
        )
    
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster sentence embeddings.
        
        Args:
            embeddings: numpy array of shape (n_sentences, embedding_dim)
            
        Returns:
            Array of cluster labels (-1 for noise)
        """
        if embeddings.size == 0:
            return np.array([])
        
        if len(embeddings.shape) != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
        
        n_sentences = embeddings.shape[0]
        logger.debug(f"Clustering {n_sentences} sentences")
        
        if n_sentences < self.min_samples:
            # Not enough sentences to form a cluster
            logger.debug("Not enough sentences for clustering, all marked as noise")
            return np.full(n_sentences, -1)
        
        # Run DBSCAN
        try:
            dbscan = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric=self.metric
            )
            labels = dbscan.fit_predict(embeddings)
            
            # Log clustering results
            unique_labels = np.unique(labels)
            cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
            noise_count = np.sum(labels == -1)
            
            logger.info(
                f"Clustering complete: {cluster_count} clusters, "
                f"{noise_count} noise points"
            )
            
            return labels
            
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            # Fallback to noise
            return np.full(n_sentences, -1)
    
    def get_cluster_stats(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about the clustering results.
        
        Args:
            labels: Array of cluster labels
            
        Returns:
            Dictionary with clustering statistics
        """
        if labels.size == 0:
            return {
                'total_sentences': 0,
                'cluster_count': 0,
                'noise_count': 0,
                'cluster_sizes': []
            }
        
        unique_labels = np.unique(labels)
        cluster_labels = [l for l in unique_labels if l != -1]
        
        cluster_sizes = []
        for label in cluster_labels:
            size = np.sum(labels == label)
            cluster_sizes.append(size)
        
        return {
            'total_sentences': len(labels),
            'cluster_count': len(cluster_labels),
            'noise_count': np.sum(labels == -1),
            'noise_percentage': (np.sum(labels == -1) / len(labels)) * 100,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0
        }
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validate embeddings before clustering.
        
        Args:
            embeddings: Embeddings to validate
            
        Returns:
            True if embeddings are valid for clustering
        """
        if embeddings.size == 0:
            logger.warning("Empty embeddings array")
            return False
        
        if len(embeddings.shape) != 2:
            logger.error(f"Invalid embeddings shape: {embeddings.shape}")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)):
            logger.error("Embeddings contain NaN values")
            return False
        
        if np.any(np.isinf(embeddings)):
            logger.error("Embeddings contain infinite values")
            return False
        
        # Check embedding dimensions are reasonable
        if embeddings.shape[1] < 10 or embeddings.shape[1] > 1000:
            logger.warning(
                f"Unusual embedding dimension: {embeddings.shape[1]}"
            )
            # Not fatal, just a warning
        
        return True


# Backward compatibility alias
ClusterAnalyzer = SklearnClusteringService


# Singleton instance for easy import
_default_cluster_analyzer: Optional[SklearnClusteringService] = None


def get_cluster_analyzer() -> SklearnClusteringService:
    """
    Get or create the default cluster analyzer instance.
    
    Returns:
        Shared SklearnClusteringService instance
    """
    global _default_cluster_analyzer
    if _default_cluster_analyzer is None:
        _default_cluster_analyzer = SklearnClusteringService()
        logger.info("Created default cluster analyzer instance")
    
    return _default_cluster_analyzer


def cluster_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Convenience function to cluster embeddings using the default analyzer.
    
    Args:
        embeddings: numpy array of embeddings
        
    Returns:
        Array of cluster labels
    """
    analyzer = get_cluster_analyzer()
    return analyzer.cluster(embeddings)
