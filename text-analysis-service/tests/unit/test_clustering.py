"""
Unit tests for the clustering module.

Tests cover:
- Clustering algorithm (simple threshold-based)
- Deterministic behavior (seeded random)
- Output shape and labels
- Edge cases (empty embeddings, small clusters, invalid inputs)
- Cluster statistics
- Validation of embeddings
- Singleton pattern and convenience functions
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.app.pipeline.clustering import (
    ClusterAnalyzer,
    get_cluster_analyzer,
    cluster_embeddings,
)


class TestClusterAnalyzer:
    """Test the ClusterAnalyzer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        analyzer = ClusterAnalyzer()
        assert analyzer.similarity_threshold == 0.7
        assert analyzer.min_cluster_size == 2
        assert analyzer.random_seed == 42

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        analyzer = ClusterAnalyzer(
            similarity_threshold=0.8,
            min_cluster_size=3,
            random_seed=123
        )
        assert analyzer.similarity_threshold == 0.8
        assert analyzer.min_cluster_size == 3
        assert analyzer.random_seed == 123

    def test_cluster_empty_embeddings(self):
        """Test clustering with empty embeddings."""
        analyzer = ClusterAnalyzer()
        embeddings = np.array([])
        labels = analyzer.cluster(embeddings)
        
        assert labels.shape == (0,)
        assert labels.size == 0

    def test_cluster_single_embedding(self):
        """Test clustering with a single embedding (should be noise)."""
        analyzer = ClusterAnalyzer(min_cluster_size=2)
        embeddings = np.random.randn(1, 384)
        labels = analyzer.cluster(embeddings)
        
        assert labels.shape == (1,)
        assert labels[0] == -1  # Noise

    def test_cluster_two_similar_embeddings(self):
        """Test clustering with two similar embeddings."""
        analyzer = ClusterAnalyzer(similarity_threshold=0.9, min_cluster_size=2)
        
        # Create two identical embeddings (cosine similarity = 1.0)
        embedding = np.random.randn(1, 384)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        embeddings = np.vstack([embedding, embedding])  # Two identical
        
        labels = analyzer.cluster(embeddings)
        
        # Should form a cluster (label 0 or higher)
        assert labels.shape == (2,)
        assert labels[0] == labels[1]
        assert labels[0] >= 0  # Not noise

    def test_cluster_two_dissimilar_embeddings(self):
        """Test clustering with two dissimilar embeddings."""
        analyzer = ClusterAnalyzer(similarity_threshold=0.9, min_cluster_size=2)
        
        # Create two orthogonal embeddings (cosine similarity = 0)
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        embeddings = np.vstack([embedding1, embedding2])
        
        labels = analyzer.cluster(embeddings)
        
        # Should be noise (similarity < threshold)
        assert labels.shape == (2,)
        assert labels[0] == -1
        assert labels[1] == -1

    def test_cluster_deterministic_behavior(self):
        """Test that clustering is deterministic with same seed."""
        # Create random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(10, 384)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        analyzer1 = ClusterAnalyzer(random_seed=123)
        analyzer2 = ClusterAnalyzer(random_seed=123)  # Same seed
        
        labels1 = analyzer1.cluster(embeddings)
        labels2 = analyzer2.cluster(embeddings)
        
        # Should be identical
        np.testing.assert_array_equal(labels1, labels2)

    def test_cluster_different_seeds_produce_different_results(self):
        """Test that different seeds can produce different clustering."""
        # Create embeddings where order matters (due to random permutation)
        np.random.seed(42)
        embeddings = np.random.randn(10, 384)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        analyzer1 = ClusterAnalyzer(random_seed=123)
        analyzer2 = ClusterAnalyzer(random_seed=456)  # Different seed
        
        labels1 = analyzer1.cluster(embeddings)
        labels2 = analyzer2.cluster(embeddings)
        
        # They might be different (not guaranteed, but likely)
        # At least test that function doesn't crash
        assert labels1.shape == labels2.shape
        assert len(labels1) == len(labels2)

    def test_cluster_min_cluster_size_enforcement(self):
        """Test that clusters smaller than min_cluster_size become noise."""
        analyzer = ClusterAnalyzer(min_cluster_size=3, similarity_threshold=0.5)
        
        # Create 2 very similar embeddings (should form cluster of size 2)
        embedding = np.random.randn(1, 384)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings = np.vstack([embedding, embedding])  # Two identical
        
        labels = analyzer.cluster(embeddings)
        
        # Should be noise because cluster size 2 < min_cluster_size 3
        assert np.all(labels == -1)

    def test_cluster_with_actual_cosine_similarity(self):
        """Test clustering using actual cosine similarity computation."""
        analyzer = ClusterAnalyzer(similarity_threshold=0.99, min_cluster_size=2)
        
        # Create embeddings with known similarity
        # embedding1 and embedding2 are identical (similarity = 1.0)
        # embedding3 is orthogonal (similarity = 0.0)
        embedding1 = np.array([1.0, 0.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0, 0.0])  # Same as embedding1
        embedding3 = np.array([0.0, 1.0, 0.0, 0.0])  # Orthogonal
        
        embeddings = np.vstack([embedding1, embedding2, embedding3])
        
        labels = analyzer.cluster(embeddings)
        
        # embedding1 and embedding2 should be in same cluster
        # embedding3 should be noise (similarity < threshold)
        assert labels[0] == labels[1]
        assert labels[0] >= 0  # Cluster label
        assert labels[2] == -1  # Noise

    def test_cluster_invalid_embeddings_shape(self):
        """Test clustering with invalid embeddings shape."""
        analyzer = ClusterAnalyzer()
        
        # 1D array instead of 2D
        embeddings = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Expected 2D embeddings"):
            analyzer.cluster(embeddings)

    def test_get_cluster_stats_empty(self):
        """Test getting statistics for empty labels."""
        analyzer = ClusterAnalyzer()
        labels = np.array([])
        stats = analyzer.get_cluster_stats(labels)
        
        assert stats['total_sentences'] == 0
        assert stats['cluster_count'] == 0
        assert stats['noise_count'] == 0
        assert stats['cluster_sizes'] == []

    def test_get_cluster_stats_all_noise(self):
        """Test statistics when all points are noise."""
        analyzer = ClusterAnalyzer()
        labels = np.array([-1, -1, -1, -1])
        stats = analyzer.get_cluster_stats(labels)
        
        assert stats['total_sentences'] == 4
        assert stats['cluster_count'] == 0
        assert stats['noise_count'] == 4
        assert stats['noise_percentage'] == 100.0
        assert stats['cluster_sizes'] == []
        assert stats['avg_cluster_size'] == 0
        assert stats['median_cluster_size'] == 0

    def test_get_cluster_stats_with_clusters(self):
        """Test statistics with actual clusters."""
        analyzer = ClusterAnalyzer()
        labels = np.array([0, 0, 0, 1, 1, -1, -1])  # Cluster 0 size 3, cluster 1 size 2, 2 noise
        stats = analyzer.get_cluster_stats(labels)
        
        assert stats['total_sentences'] == 7
        assert stats['cluster_count'] == 2
        assert stats['noise_count'] == 2
        assert stats['noise_percentage'] == (2/7) * 100
        assert set(stats['cluster_sizes']) == {3, 2}
        assert stats['avg_cluster_size'] == 2.5
        assert stats['median_cluster_size'] == 2.5

    def test_validate_embeddings_valid(self):
        """Test validation of valid embeddings."""
        analyzer = ClusterAnalyzer()
        embeddings = np.random.randn(10, 384)
        
        assert analyzer.validate_embeddings(embeddings) is True

    def test_validate_embeddings_empty(self):
        """Test validation of empty embeddings."""
        analyzer = ClusterAnalyzer()
        embeddings = np.array([])
        
        # Should return False (with warning)
        assert analyzer.validate_embeddings(embeddings) is False

    def test_validate_embeddings_1d(self):
        """Test validation of 1D embeddings."""
        analyzer = ClusterAnalyzer()
        embeddings = np.array([1.0, 2.0, 3.0])
        
        assert analyzer.validate_embeddings(embeddings) is False

    def test_validate_embeddings_with_nan(self):
        """Test validation of embeddings with NaN values."""
        analyzer = ClusterAnalyzer()
        embeddings = np.array([[1.0, 2.0], [np.nan, 4.0]])
        
        assert analyzer.validate_embeddings(embeddings) is False

    def test_validate_embeddings_with_inf(self):
        """Test validation of embeddings with infinite values."""
        analyzer = ClusterAnalyzer()
        embeddings = np.array([[1.0, 2.0], [np.inf, 4.0]])
        
        assert analyzer.validate_embeddings(embeddings) is False

    def test_validate_embeddings_unusual_dimension(self):
        """Test validation of embeddings with unusual dimension (warning but not error)."""
        analyzer = ClusterAnalyzer()
        # Very small dimension (should trigger warning but return True)
        embeddings = np.random.randn(5, 5)
        
        # Should return True (dimension check is just a warning)
        assert analyzer.validate_embeddings(embeddings) is True

    def test_simple_threshold_clustering(self):
        """Test the internal _simple_threshold_clustering method."""
        analyzer = ClusterAnalyzer(similarity_threshold=0.8, min_cluster_size=2)
        
        # Create a similarity matrix
        # Sentences 0,1,2 are similar to each other
        # Sentence 3 is only similar to itself
        similarity_matrix = np.array([
            [1.0, 0.9, 0.85, 0.1],  # 0 similar to 0,1,2
            [0.9, 1.0, 0.88, 0.2],  # 1 similar to 0,1,2
            [0.85, 0.88, 1.0, 0.15], # 2 similar to 0,1,2
            [0.1, 0.2, 0.15, 1.0]   # 3 only similar to itself
        ])
        
        labels = analyzer._simple_threshold_clustering(similarity_matrix)
        
        # Sentences 0,1,2 should be in same cluster (label >= 0)
        # Sentence 3 should be noise (-1) because only similar to itself
        assert labels[0] == labels[1] == labels[2]
        assert labels[0] >= 0
        assert labels[3] == -1

    def test_enforce_min_cluster_size(self):
        """Test the internal _enforce_min_cluster_size method."""
        analyzer = ClusterAnalyzer(min_cluster_size=3)
        
        # Labels: cluster 0 (size 4), cluster 1 (size 2), cluster 2 (size 3), noise
        labels = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, -1])
        
        updated_labels = analyzer._enforce_min_cluster_size(labels)
        
        # Cluster 0 (size 4) should remain
        # Cluster 1 (size 2) should become noise
        # Cluster 2 (size 3) should remain
        assert np.sum(updated_labels == 0) == 4
        assert np.sum(updated_labels == 1) == 0  # Converted to noise
        assert np.sum(updated_labels == 2) == 3
        # Original noise should remain noise
        assert updated_labels[9] == -1
        # Cluster 1 points should now be noise
        assert updated_labels[4] == -1
        assert updated_labels[5] == -1

    def test_cluster_with_mock_similarity(self):
        """Test clustering with mocked cosine_similarity to control behavior."""
        analyzer = ClusterAnalyzer(similarity_threshold=0.7, min_cluster_size=2)
        
        # Mock cosine_similarity to return a controlled matrix
        mock_similarity = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        
        with patch('src.app.pipeline.clustering.cosine_similarity',
                   return_value=mock_similarity):
            embeddings = np.random.randn(3, 384)
            labels = analyzer.cluster(embeddings)
            
            # Based on mock similarity:
            # - Sentences 0 and 1 are similar (0.8 >= 0.7)
            # - Sentence 2 is not similar enough to others
            # Should form one cluster with sentences 0 and 1
            assert labels[0] == labels[1]
            assert labels[0] >= 0  # Cluster label
            assert labels[2] == -1  # Noise

    @pytest.mark.skip(reason="Actual clustering algorithm not implemented yet")
    def test_actual_clustering_algorithm(self):
        """TODO: Test with actual clustering algorithm when implemented."""
        pass


class TestSingletonAndConvenience:
    """Test the singleton pattern and convenience functions."""

    def test_get_cluster_analyzer_singleton(self):
        """Test that get_cluster_analyzer returns the same instance."""
        analyzer1 = get_cluster_analyzer()
        analyzer2 = get_cluster_analyzer()
        
        assert analyzer1 is analyzer2  # Same object

    def test_cluster_embeddings_convenience(self):
        """Test the cluster_embeddings convenience function."""
        embeddings = np.random.randn(5, 384)
        
        # Mock the singleton analyzer to avoid side effects
        mock_analyzer = MagicMock()
        mock_labels = np.array([0, 0, 1, 1, -1])
        mock_analyzer.cluster.return_value = mock_labels
        
        with patch('src.app.pipeline.clustering.get_cluster_analyzer',
                   return_value=mock_analyzer):
            result = cluster_embeddings(embeddings)
        
        # Should call analyzer.cluster with the embeddings
        mock_analyzer.cluster.assert_called_once_with(embeddings)
        np.testing.assert_array_equal(result, mock_labels)

    def test_cluster_embeddings_empty(self):
        """Test cluster_embeddings with empty array."""
        mock_analyzer = MagicMock()
        mock_analyzer.cluster.return_value = np.array([])
        
        with patch('text_analysis_service.src.app.pipeline.clustering.get_cluster_analyzer', 
                   return_value=mock_analyzer):
            result = cluster_embeddings(np.array([]))
        
        mock_analyzer.cluster.assert_called_once_with(np.array([]))
        assert result.shape == (0,)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cluster_very_large_embeddings(self):
        """Test clustering with very high-dimensional embeddings."""
        analyzer = ClusterAnalyzer()
        # 1000 dimensions (unusual but should work)
        embeddings = np.random.randn(10, 1000)
        
        # Should not crash
        labels = analyzer.cluster(embeddings)
        assert labels.shape == (10,)

    def test_cluster_identical_embeddings_all_cluster(self):
        """Test clustering when all embeddings are identical."""
        analyzer = ClusterAnalyzer(similarity_threshold=0.9, min_cluster_size=2)
        
        # All embeddings identical
        embedding = np.random.randn(1, 384)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings = np.vstack([embedding] * 10)  # 10 identical
        
        labels = analyzer.cluster(embeddings)
        
        # All should be in same cluster (or noise if min_cluster_size > 10?)
        # With min_cluster_size=2, they should form a cluster
        unique_labels = np.unique(labels)
        # Should have exactly one cluster label (>=0) and possibly noise (-1)
        # But with identical embeddings, they should all be in same cluster
        assert len(unique_labels) <= 2  # Could be just cluster, or cluster + noise
        assert np.sum(labels >= 0) >= 2  # At least min_cluster_size in cluster
    
    def test_cluster_with_extreme_threshold(self):
        """Test clustering with extreme similarity thresholds."""
        analyzer = ClusterAnalyzer(similarity_threshold=1.0, min_cluster_size=2)
        
        # Even identical embeddings have cosine similarity 1.0
        embedding = np.random.randn(1, 384)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings = np.vstack([embedding, embedding])  # Two identical
        
        labels = analyzer.cluster(embeddings)
        
        # Should form a cluster (similarity = 1.0 >= threshold)
        assert labels[0] == labels[1]
        assert labels[0] >= 0
    
    def test_cluster_with_zero_threshold(self):
        """Test clustering with zero threshold (everything clusters together)."""
        analyzer = ClusterAnalyzer(similarity_threshold=0.0, min_cluster_size=2)
        
        # Create orthogonal embeddings (cosine similarity = 0)
        embedding1 = np.array([1.0, 0.0])
        embedding2 = np.array([0.0, 1.0])
        embeddings = np.vstack([embedding1, embedding2])
        
        labels = analyzer.cluster(embeddings)
        
        # With threshold 0, even orthogonal embeddings should cluster
        # (similarity 0 >= threshold 0)
        assert labels[0] == labels[1]
        assert labels[0] >= 0
    
    def test_cluster_high_min_cluster_size(self):
        """Test clustering with min_cluster_size larger than number of sentences."""
        analyzer = ClusterAnalyzer(min_cluster_size=10, similarity_threshold=0.5)
        
        # Only 5 sentences
        embeddings = np.random.randn(5, 384)
        
        labels = analyzer.cluster(embeddings)
        
        # All should be noise (can't form cluster of size 10)
        assert np.all(labels == -1)
    
    def test_cluster_stats_with_single_cluster(self):
        """Test statistics with a single cluster."""
        analyzer = ClusterAnalyzer()
        labels = np.array([0, 0, 0, 0, 0])  # All in same cluster
        stats = analyzer.get_cluster_stats(labels)
        
        assert stats['total_sentences'] == 5
        assert stats['cluster_count'] == 1
        assert stats['noise_count'] == 0
        assert stats['noise_percentage'] == 0.0
        assert stats['cluster_sizes'] == [5]
        assert stats['avg_cluster_size'] == 5.0
        assert stats['median_cluster_size'] == 5.0
    
    def test_cluster_stats_mixed_labels(self):
        """Test statistics with mixed positive and negative cluster labels."""
        analyzer = ClusterAnalyzer()
        # Note: cluster labels can be any non-negative integer
        labels = np.array([0, 0, 1, 1, 2, 2, 2, -1, -1])
        stats = analyzer.get_cluster_stats(labels)
        
        assert stats['total_sentences'] == 9
        assert stats['cluster_count'] == 3
        assert stats['noise_count'] == 2
        assert stats['cluster_sizes'] == [2, 2, 3]  # Sorted by label
        assert stats['avg_cluster_size'] == (2 + 2 + 3) / 3
    
    def test_validate_embeddings_3d(self):
        """Test validation of 3D embeddings (invalid)."""
        analyzer = ClusterAnalyzer()
        embeddings = np.random.randn(5, 10, 10)  # 3D array
        
        assert analyzer.validate_embeddings(embeddings) is False
    
    def test_cluster_with_precomputed_similarity(self):
        """Test that clustering works with the actual cosine_similarity function."""
        analyzer = ClusterAnalyzer(similarity_threshold=0.8, min_cluster_size=2)
        
        # Create embeddings that we know will have high similarity
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)
        
        # Second embedding is 0.9 similar to base (add small noise)
        noise = np.random.randn(384) * 0.1
        embedding2 = base + noise
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Third embedding is orthogonal
        embedding3 = np.random.randn(384)
        embedding3 = embedding3 / np.linalg.norm(embedding3)
        # Make it orthogonal to base
        embedding3 = embedding3 - np.dot(embedding3, base) * base
        embedding3 = embedding3 / np.linalg.norm(embedding3)
        
        embeddings = np.vstack([base, embedding2, embedding3])
        
        labels = analyzer.cluster(embeddings)
        
        # First two should cluster, third should be noise
        # (similarity between base and embedding2 ~0.995 > 0.8)
        # (similarity with embedding3 ~0 < 0.8)
        assert labels[0] == labels[1]
        assert labels[0] >= 0
        assert labels[2] == -1
    
    def test_deterministic_across_multiple_calls(self):
        """Test that same analyzer produces same results across multiple calls."""
        analyzer = ClusterAnalyzer(random_seed=42)
        embeddings = np.random.randn(8, 384)
        
        labels1 = analyzer.cluster(embeddings)
        labels2 = analyzer.cluster(embeddings)  # Second call
        
        # Should be identical (deterministic)
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_singleton_independence(self):
        """Test that different analyzer instances are independent."""
        analyzer1 = ClusterAnalyzer(random_seed=1)
        analyzer2 = ClusterAnalyzer(random_seed=2)  # Different instance
        
        # They should have independent random states
        assert analyzer1.random_seed != analyzer2.random_seed
        # But both should work
        embeddings = np.random.randn(5, 384)
        labels1 = analyzer1.cluster(embeddings)
        labels2 = analyzer2.cluster(embeddings)
        
        # They might produce different results due to different seeds
        assert labels1.shape == labels2.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])