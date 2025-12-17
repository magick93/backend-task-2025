"""
Comparison module for comparative analysis.

Post-process clusters for comparison:
- Baseline vs comparison
- Similarities and differences
- Do not redo clustering here
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ComparisonAnalyzer:
    """
    Analyze similarities and differences between baseline and comparison clusters.
    
    This module post-processes already-clustered data to identify:
    - Similar clusters between baseline and comparison
    - Unique clusters in each dataset
    - Key differences in sentiment, size, or content
    - Overall similarity score
    
    Does NOT redo clustering - works with existing cluster assignments.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_cluster_size: int = 2,
        max_comparisons: int = 10
    ):
        """
        Initialize the comparison analyzer.
        
        Args:
            similarity_threshold: Threshold for considering clusters similar (0-1)
            min_cluster_size: Minimum cluster size to consider for comparison
            max_comparisons: Maximum number of similarity pairs to return
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_comparisons = max_comparisons
        
        logger.debug(
            f"ComparisonAnalyzer initialized with "
            f"similarity_threshold={similarity_threshold}, "
            f"min_cluster_size={min_cluster_size}"
        )
    
    def compare_clusters(
        self,
        baseline_clusters: List[Dict[str, Any]],
        comparison_clusters: List[Dict[str, Any]],
        baseline_embeddings: Optional[np.ndarray] = None,
        comparison_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare baseline and comparison clusters.
        
        Args:
            baseline_clusters: List of cluster insights from baseline dataset
            comparison_clusters: List of cluster insights from comparison dataset
            baseline_embeddings: Optional embeddings for baseline sentences
            comparison_embeddings: Optional embeddings for comparison sentences
            
        Returns:
            Dictionary with comparison results including:
            - similarities: List of similar cluster pairs
            - differences: List of unique clusters in each dataset
            - similarity_score: Overall similarity score (0-1)
            - summary: Text summary of comparison
        """
        logger.info(
            f"Comparing {len(baseline_clusters)} baseline clusters "
            f"with {len(comparison_clusters)} comparison clusters"
        )
        
        # Filter out small clusters
        baseline_filtered = self._filter_clusters(baseline_clusters)
        comparison_filtered = self._filter_clusters(comparison_clusters)
        
        # Calculate cluster centroids if embeddings provided
        baseline_centroids = self._calculate_centroids(baseline_filtered, baseline_embeddings)
        comparison_centroids = self._calculate_centroids(comparison_filtered, comparison_embeddings)
        
        # Find similar clusters
        similarities = self._find_similar_clusters(
            baseline_filtered, comparison_filtered,
            baseline_centroids, comparison_centroids
        )
        
        # Find unique clusters (differences)
        differences = self._find_unique_clusters(
            baseline_filtered, comparison_filtered, similarities
        )
        
        # Calculate overall similarity score
        similarity_score = self._calculate_similarity_score(
            baseline_filtered, comparison_filtered, similarities
        )
        
        # Generate summary
        summary = self._generate_comparison_summary(
            baseline_filtered, comparison_filtered,
            similarities, differences, similarity_score
        )
        
        return {
            'similarities': similarities[:self.max_comparisons],
            'differences': differences,
            'similarity_score': similarity_score,
            'summary': summary,
            'baseline_cluster_count': len(baseline_filtered),
            'comparison_cluster_count': len(comparison_filtered),
            'similarity_pair_count': len(similarities)
        }
    
    def _filter_clusters(
        self, 
        clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter clusters based on size and quality.
        
        Args:
            clusters: List of cluster insights
            
        Returns:
            Filtered list of clusters
        """
        filtered = []
        for cluster in clusters:
            size = cluster.get('sentence_count', 0)
            if size >= self.min_cluster_size:
                filtered.append(cluster)
        
        logger.debug(f"Filtered clusters: {len(clusters)} -> {len(filtered)}")
        return filtered
    
    def _calculate_centroids(
        self,
        clusters: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Calculate centroids for clusters.
        
        Args:
            clusters: List of cluster insights
            embeddings: Optional embeddings for sentences
            
        Returns:
            Dictionary mapping cluster_id to centroid vector
        """
        centroids = {}
        
        if embeddings is None:
            # Return empty centroids if no embeddings provided
            return centroids
        
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id')
            # TODO: In a real implementation, we would need sentence indices
            # to calculate actual centroids. For now, return empty.
            pass
        
        return centroids
    
    def _find_similar_clusters(
        self,
        baseline_clusters: List[Dict[str, Any]],
        comparison_clusters: List[Dict[str, Any]],
        baseline_centroids: Dict[int, np.ndarray],
        comparison_centroids: Dict[int, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Find similar clusters between baseline and comparison.
        
        Args:
            baseline_clusters: Filtered baseline clusters
            comparison_clusters: Filtered comparison clusters
            baseline_centroids: Baseline cluster centroids
            comparison_centroids: Comparison cluster centroids
            
        Returns:
            List of similar cluster pairs with similarity scores
        """
        similarities = []
        
        # Simple heuristic-based similarity (placeholder)
        # In a real implementation, this would use embedding similarity
        
        for b_cluster in baseline_clusters:
            for c_cluster in comparison_clusters:
                # Calculate similarity score using multiple factors
                similarity = self._calculate_cluster_similarity(b_cluster, c_cluster)
                
                if similarity >= self.similarity_threshold:
                    similarities.append({
                        'baseline_cluster': {
                            'id': b_cluster.get('cluster_id'),
                            'title': b_cluster.get('title', 'Unknown'),
                            'size': b_cluster.get('sentence_count', 0),
                            'sentiment': b_cluster.get('sentiment', 'neutral'),
                            'key_terms': b_cluster.get('key_terms', [])[:5]
                        },
                        'comparison_cluster': {
                            'id': c_cluster.get('cluster_id'),
                            'title': c_cluster.get('title', 'Unknown'),
                            'size': c_cluster.get('sentence_count', 0),
                            'sentiment': c_cluster.get('sentiment', 'neutral'),
                            'key_terms': c_cluster.get('key_terms', [])[:5]
                        },
                        'similarity_score': similarity,
                        'similarity_type': self._determine_similarity_type(
                            b_cluster, c_cluster, similarity
                        )
                    })
        
        # Sort by similarity score (highest first)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.debug(f"Found {len(similarities)} similar cluster pairs")
        return similarities
    
    def _calculate_cluster_similarity(
        self,
        cluster1: Dict[str, Any],
        cluster2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity score between two clusters.
        
        Uses multiple heuristics:
        1. Title similarity (if available)
        2. Key term overlap
        3. Sentiment match
        4. Size ratio
        
        Args:
            cluster1: First cluster
            cluster2: Second cluster
            
        Returns:
            Similarity score (0-1)
        """
        score = 0.0
        factors = 0
        
        # 1. Title similarity (simple string matching)
        title1 = cluster1.get('title', '').lower()
        title2 = cluster2.get('title', '').lower()
        
        if title1 and title2:
            # Simple word overlap
            words1 = set(title1.split())
            words2 = set(title2.split())
            if words1 and words2:
                overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
                score += overlap * 0.4  # Weight: 40%
                factors += 0.4
        
        # 2. Key term overlap
        terms1 = set(cluster1.get('key_terms', []))
        terms2 = set(cluster2.get('key_terms', []))
        
        if terms1 and terms2:
            overlap = len(terms1.intersection(terms2)) / max(len(terms1), len(terms2))
            score += overlap * 0.3  # Weight: 30%
            factors += 0.3
        
        # 3. Sentiment match
        sentiment1 = cluster1.get('sentiment', 'neutral')
        sentiment2 = cluster2.get('sentiment', 'neutral')
        
        if sentiment1 == sentiment2:
            score += 0.2  # Weight: 20%
        elif sentiment1 in ['positive', 'negative'] and sentiment2 in ['positive', 'negative']:
            # Both have strong sentiment but opposite
            score += 0.05  # Small weight for at least having strong sentiment
        factors += 0.2
        
        # 4. Size ratio (clusters of similar size are more comparable)
        size1 = cluster1.get('sentence_count', 1)
        size2 = cluster2.get('sentence_count', 1)
        size_ratio = min(size1, size2) / max(size1, size2)
        score += size_ratio * 0.1  # Weight: 10%
        factors += 0.1
        
        # Normalize by total factor weight
        if factors > 0:
            score = score / factors
        
        return min(max(score, 0.0), 1.0)
    
    def _determine_similarity_type(
        self,
        cluster1: Dict[str, Any],
        cluster2: Dict[str, Any],
        similarity_score: float
    ) -> str:
        """
        Determine the type of similarity between clusters.
        
        Args:
            cluster1: First cluster
            cluster2: Second cluster
            similarity_score: Calculated similarity score
            
        Returns:
            Similarity type description
        """
        if similarity_score >= 0.9:
            return "Very similar topics"
        elif similarity_score >= 0.8:
            return "Similar topics with variations"
        elif similarity_score >= 0.7:
            return "Related topics"
        elif similarity_score >= 0.6:
            return "Partially related topics"
        else:
            return "Weakly related topics"
    
    def _find_unique_clusters(
        self,
        baseline_clusters: List[Dict[str, Any]],
        comparison_clusters: List[Dict[str, Any]],
        similarities: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find clusters unique to each dataset.
        
        Args:
            baseline_clusters: Filtered baseline clusters
            comparison_clusters: Filtered comparison clusters
            similarities: List of similar cluster pairs
            
        Returns:
            Dictionary with 'baseline_unique' and 'comparison_unique' lists
        """
        # Extract cluster IDs from similarities
        similar_baseline_ids = {
            pair['baseline_cluster']['id']
            for pair in similarities
            if 'baseline_cluster' in pair
        }
        
        similar_comparison_ids = {
            pair['comparison_cluster']['id']
            for pair in similarities
            if 'comparison_cluster' in pair
        }
        
        # Find unique clusters
        baseline_unique = [
            cluster for cluster in baseline_clusters
            if cluster.get('cluster_id') not in similar_baseline_ids
        ]
        
        comparison_unique = [
            cluster for cluster in comparison_clusters
            if cluster.get('cluster_id') not in similar_comparison_ids
        ]
        
        # Format unique clusters
        formatted_baseline = [
            {
                'id': cluster.get('cluster_id'),
                'title': cluster.get('title', 'Unknown'),
                'size': cluster.get('sentence_count', 0),
                'sentiment': cluster.get('sentiment', 'neutral'),
                'key_terms': cluster.get('key_terms', [])[:5],
                'representative_sentence': (
                    cluster.get('representative_sentences', [])[0]
                    if cluster.get('representative_sentences')
                    else ''
                )
            }
            for cluster in baseline_unique
        ]
        
        formatted_comparison = [
            {
                'id': cluster.get('cluster_id'),
                'title': cluster.get('title', 'Unknown'),
                'size': cluster.get('sentence_count', 0),
                'sentiment': cluster.get('sentiment', 'neutral'),
                'key_terms': cluster.get('key_terms', [])[:5],
                'representative_sentence': (
                    cluster.get('representative_sentences', [])[0]
                    if cluster.get('representative_sentences')
                    else ''
                )
            }
            for cluster in comparison_unique
        ]
        
        logger.debug(
            f"Found {len(formatted_baseline)} unique baseline clusters, "
            f"{len(formatted_comparison)} unique comparison clusters"
        )
        
        return {
            'baseline_unique': formatted_baseline,
            'comparison_unique': formatted_comparison
        }
    
    def _calculate_similarity_score(
        self,
        baseline_clusters: List[Dict[str, Any]],
        comparison_clusters: List[Dict[str, Any]],
        similarities: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall similarity score between datasets.
        
        Args:
            baseline_clusters: Filtered baseline clusters
            comparison_clusters: Filtered comparison clusters
            similarities: List of similar cluster pairs
            
        Returns:
            Overall similarity score (0-1)
        """
        if not baseline_clusters or not comparison_clusters:
            return 0.0
        
        # Calculate coverage: how many clusters are matched
        total_clusters = len(baseline_clusters) + len(comparison_clusters)
        
        if total_clusters == 0:
            return 0.0
        
        # Count matched clusters (each similarity matches 2 clusters)
        matched_clusters = min(
            len(similarities) * 2,
            len(baseline_clusters) + len(comparison_clusters)
        )
        
        # Coverage ratio
        coverage = matched_clusters / total_clusters
        
        # Average similarity of matched pairs
        if similarities:
            avg_similarity = np.mean([p['similarity_score'] for p in similarities])
        else:
            avg_similarity = 0.0
        
        # Combined score: weighted average of coverage and similarity
        # Give more weight to coverage (60%) than average similarity (40%)
        combined_score = (coverage * 0.6) + (avg_similarity * 0.4)
        
        return min(max(combined_score, 0.0), 1.0)
    
    def _generate_comparison_summary(
        self,
        baseline_clusters: List[Dict[str, Any]],
        comparison_clusters: List[Dict[str, Any]],
        similarities: List[Dict[str, Any]],
        differences: Dict[str, List[Dict[str, Any]]],
        similarity_score: float
    ) -> str:
        """
        Generate a text summary of the comparison.
        
        Args:
            baseline_clusters: Filtered baseline clusters
            comparison_clusters: Filtered comparison clusters
            similarities: List of similar cluster pairs
            differences: Dictionary of unique clusters
            similarity_score: Overall similarity score
            
        Returns:
            Text summary
        """
        baseline_count = len(baseline_clusters)
        comparison_count = len(comparison_clusters)
        similarity_count = len(similarities)
        baseline_unique = len(differences.get('baseline_unique', []))
        comparison_unique = len(differences.get('comparison_unique', []))
        
        # Determine similarity level
        if similarity_score >= 0.8:
            similarity_level = "very similar"
        elif similarity_score >= 0.6:
            similarity_level = "moderately similar"
        elif similarity_score >= 0.4:
            similarity_level = "somewhat different"
        else:
            similarity_level = "very different"
        
        # Generate summary based on counts
        if similarity_count == 0:
            summary = (
                f"The baseline and comparison datasets are {similarity_level}. "
                f"No strong similarities were found between the {baseline_count} "
                f"baseline topics and {comparison_count} comparison topics."
            )
        elif similarity_count == 1:
            summary = (
                f"The datasets are {similarity_level} with 1 shared topic. "
                f"Baseline has {baseline_unique} unique topics, "
                f"comparison has {comparison_unique} unique topics."
            )
        else:
            summary = (
                f"The datasets are {similarity_level} with {similarity_count} "
                f"shared topics. Baseline has {baseline_unique} unique topics, "
                f"comparison has {comparison_unique} unique topics."
            )
        
        # Add sentiment comparison if available
        baseline_sentiments = [c.get('sentiment', 'neutral') for c in baseline_clusters]
        comparison_sentiments = [c.get('sentiment', 'neutral') for c in comparison_clusters]
        
        if baseline_sentiments and comparison_sentiments:
            baseline_pos = baseline_sentiments.count('positive') / len(baseline_sentiments)
            comparison_pos = comparison_sentiments.count('positive') / len(comparison_sentiments)
            
            if abs(baseline_pos - comparison_pos) > 0.3:
                if comparison_pos > baseline_pos:
                    summary += " The comparison dataset shows more positive sentiment."
                else:
                    summary += " The baseline dataset shows more positive sentiment."
        
        return summary
    
    def get_similarity_description(self, similarity_score: float) -> str:
        """
        Get a human-readable description of similarity score.
        
        Args:
            similarity_score: Overall similarity score (0-1)
            
        Returns:
            Description of similarity level
        """
        if similarity_score >= 0.9:
            return "Very high similarity - datasets are nearly identical in topics"
        elif similarity_score >= 0.8:
            return "High similarity - datasets share most topics"
        elif similarity_score >= 0.7:
            return "Moderate similarity - datasets have significant overlap"
        elif similarity_score >= 0.6:
            return "Some similarity - datasets share some topics"
        elif similarity_score >= 0.4:
            return "Low similarity - datasets have limited overlap"
        else:
            return "Very low similarity - datasets are largely different"


# Singleton instance for easy import
_default_comparison_analyzer: Optional[ComparisonAnalyzer] = None


def get_comparison_analyzer() -> ComparisonAnalyzer:
    """
    Get or create the default comparison analyzer instance.
    
    Returns:
        Shared ComparisonAnalyzer instance
    """
    global _default_comparison_analyzer
    if _default_comparison_analyzer is None:
        _default_comparison_analyzer = ComparisonAnalyzer()
        logger.info("Created default comparison analyzer instance")
    
    return _default_comparison_analyzer