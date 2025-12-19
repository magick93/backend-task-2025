"""
Unit tests for the clustering module.

Tests cover:
- Clustering algorithm (DBSCAN)
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
        assert analyzer.eps == 0.3
        assert analyzer.min_samples == 2
        assert analyzer.metric == 'cosine'

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        analyzer = ClusterAnalyzer(
            eps=0.5,
            min_samples=3,
            metric='euclidean'
        )
        assert analyzer.eps == 0.5
        assert analyzer.min_samples == 3
        assert analyzer.metric == 'euclidean'

    def test_cluster_empty_embeddings(self):
        """Test clustering with empty embeddings."""
        analyzer = ClusterAnalyzer()
        embeddings = np.array([])
        labels = analyzer.cluster(embeddings)
        
        assert labels.shape == (0,)
        assert labels.size == 0

    def test_cluster_single_embedding(self):
        """Test clustering with a single embedding (should be noise)."""
        analyzer = ClusterAnalyzer(min_samples=2)
        embeddings = np.random.randn(1, 384)
        labels = analyzer.cluster(embeddings)
        
        # Should return noise because n_sentences < min_samples
        assert labels.shape == (1,)
        assert labels[0] == -1

    def test_cluster_two_similar_embeddings(self):
        """Test clustering with two similar embeddings."""
        analyzer = ClusterAnalyzer(eps=0.5, min_samples=2, metric='cosine')
        
        # Create two identical embeddings (cosine distance = 0)
        embedding = np.random.randn(1, 384)
        embeddings = np.vstack([embedding, embedding])  # Two identical
        
        labels = analyzer.cluster(embeddings)
        
        # Should form a cluster (label 0 or higher)
        assert labels.shape == (2,)
        assert labels[0] == labels[1]
        assert labels[0] != -1  # Not noise

    def test_cluster_two_dissimilar_embeddings(self):
        """Test clustering with two dissimilar embeddings."""
        analyzer = ClusterAnalyzer(eps=0.1, min_samples=2, metric='cosine')
        
        # Create two orthogonal embeddings (cosine distance = 1.0)
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        embeddings = np.vstack([embedding1, embedding2])
        
        labels = analyzer.cluster(embeddings)
        
        # Should be noise (distance > eps)
        assert labels.shape == (2,)
        assert labels[0] == -1
        assert labels[1] == -1

    def test_cluster_min_samples_enforcement(self):
        """Test that min_samples is enforced."""
        # Need 3 samples to form a cluster
        analyzer = ClusterAnalyzer(min_samples=3, eps=0.5)
        
        # Create 2 very similar embeddings
        embedding = np.random.randn(1, 384)
        embeddings = np.vstack([embedding, embedding])
        
        labels = analyzer.cluster(embeddings)
        
        # Should be noise because we have 2 samples but need 3
        # Logic in code: if n_sentences < min_samples, returns noise immediately.
        # But if we had more sentences but only 2 were close, DBSCAN would also mark them as noise.
        assert np.all(labels == -1)

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

    def test_validate_embeddings_3d(self):
        """Test validation of 3D embeddings (invalid)."""
        analyzer = ClusterAnalyzer()
        embeddings = np.random.randn(5, 10, 10)  # 3D array
        
        assert analyzer.validate_embeddings(embeddings) is False

    def test_cluster_with_mock_dbscan(self):
        """Test clustering with mocked DBSCAN to verify integration."""
        analyzer = ClusterAnalyzer(eps=0.5, min_samples=2)
        embeddings = np.random.randn(5, 384)
        
        with patch('src.app.pipeline.clustering.DBSCAN') as mock_dbscan_cls:
            mock_dbscan = mock_dbscan_cls.return_value
            mock_dbscan.fit_predict.return_value = np.array([0, 0, 0, -1, -1])
            
            labels = analyzer.cluster(embeddings)
            
            # Verify DBSCAN was initialized correctly
            mock_dbscan_cls.assert_called_with(eps=0.5, min_samples=2, metric='cosine')
            
            # Verify fit_predict was called
            mock_dbscan.fit_predict.assert_called_once_with(embeddings)
            
            # Verify output
            np.testing.assert_array_equal(labels, np.array([0, 0, 0, -1, -1]))

    def test_cluster_dbscan_exception(self):
        """Test handling of DBSCAN exception."""
        analyzer = ClusterAnalyzer()
        embeddings = np.random.randn(5, 384)
        
        with patch('src.app.pipeline.clustering.DBSCAN') as mock_dbscan_cls:
            mock_dbscan = mock_dbscan_cls.return_value
            mock_dbscan.fit_predict.side_effect = Exception("DBSCAN failed")
            
            labels = analyzer.cluster(embeddings)
            
            # Should fallback to all noise
            assert np.all(labels == -1)
            assert len(labels) == 5

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
        analyzer = ClusterAnalyzer(eps=0.1, min_samples=2)
        
        # All embeddings identical
        embedding = np.random.randn(1, 384)
        embeddings = np.vstack([embedding] * 10)  # 10 identical
        
        labels = analyzer.cluster(embeddings)
        
        # All should be in same cluster
        unique_labels = np.unique(labels)
        # Should have exactly one cluster label (>=0)
        assert len(unique_labels) == 1
        assert unique_labels[0] != -1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
