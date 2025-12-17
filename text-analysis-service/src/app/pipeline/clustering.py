"""
Clustering module for sentence embeddings.

Implement placeholder clustering logic:
- Use cosine similarity
- Keep deterministic behavior (seeded)
- Focus on clarity, not quality
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ClusterAnalyzer:
    """
    Clustering analyzer for sentence embeddings.
    
    Uses cosine similarity and simple deterministic clustering.
    This is a placeholder implementation - focus is on clarity and
    deterministic behavior rather than production-quality clustering.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_cluster_size: int = 2,
        random_seed: int = 42
    ):
        """
        Initialize the cluster analyzer.
        
        Args:
            similarity_threshold: Minimum cosine similarity for clustering
            min_cluster_size: Minimum sentences per cluster (smaller become noise)
            random_seed: Seed for deterministic behavior
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.random_seed = random_seed
        
        logger.debug(
            f"ClusterAnalyzer initialized: threshold={similarity_threshold}, "
            f"min_size={min_cluster_size}, seed={random_seed}"
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
        
        # Placeholder implementation: simple deterministic clustering
        # TODO: Replace with actual clustering algorithm (DBSCAN, HDBSCAN, etc.)
        
        if n_sentences < self.min_cluster_size:
            # Not enough sentences to form clusters
            logger.debug("Not enough sentences for clustering, all marked as noise")
            return np.full(n_sentences, -1)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Simple clustering based on similarity threshold
        labels = self._simple_threshold_clustering(similarity_matrix)
        
        # Enforce minimum cluster size
        labels = self._enforce_min_cluster_size(labels)
        
        # Log clustering results
        unique_labels = np.unique(labels)
        cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = np.sum(labels == -1)
        
        logger.info(
            f"Clustering complete: {cluster_count} clusters, "
            f"{noise_count} noise points"
        )
        
        return labels
    
    def _simple_threshold_clustering(
        self, 
        similarity_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Simple threshold-based clustering.
        
        Args:
            similarity_matrix: Pairwise cosine similarity matrix
            
        Returns:
            Array of cluster labels
        """
        n = similarity_matrix.shape[0]
        labels = np.full(n, -1)  # Start with all as noise
        current_label = 0
        
        # Create visited array
        visited = np.zeros(n, dtype=bool)
        
        # Set random seed for deterministic behavior
        np.random.seed(self.random_seed)
        
        # Process sentences in random but deterministic order
        order = np.random.permutation(n)
        
        for i in order:
            if visited[i]:
                continue
                
            # Find similar sentences
            similar_indices = np.where(
                similarity_matrix[i] >= self.similarity_threshold
            )[0]
            
            if len(similar_indices) >= self.min_cluster_size:
                # Found a cluster
                labels[similar_indices] = current_label
                visited[similar_indices] = True
                current_label += 1
            else:
                # Mark as noise
                labels[i] = -1
                visited[i] = True
        
        return labels
    
    def _enforce_min_cluster_size(self, labels: np.ndarray) -> np.ndarray:
        """
        Enforce minimum cluster size by converting small clusters to noise.
        
        Args:
            labels: Array of cluster labels
            
        Returns:
            Updated labels with small clusters converted to noise
        """
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:
                continue
                
            cluster_size = np.sum(labels == label)
            if cluster_size < self.min_cluster_size:
                # Convert small cluster to noise
                labels[labels == label] = -1
                logger.debug(f"Converted cluster {label} to noise (size: {cluster_size})")
        
        return labels
    
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


# Singleton instance for easy import
_default_cluster_analyzer: Optional[ClusterAnalyzer] = None


def get_cluster_analyzer() -> ClusterAnalyzer:
    """
    Get or create the default cluster analyzer instance.
    
    Returns:
        Shared ClusterAnalyzer instance
    """
    global _default_cluster_analyzer
    if _default_cluster_analyzer is None:
        _default_cluster_analyzer = ClusterAnalyzer()
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