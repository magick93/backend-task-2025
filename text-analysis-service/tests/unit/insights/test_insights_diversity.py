"""
Tests for diversity score calculation.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.app.pipeline.insights import InsightGenerator


class TestDiversityCalculation:
    """Test _calculate_diversity method."""
    
    def test_diversity_single_sentence(self):
        """Test diversity calculation for single sentence."""
        generator = InsightGenerator()
        embeddings = np.array([[1.0, 2.0, 3.0]])
        
        diversity = generator._calculate_diversity(embeddings)
        
        # Single sentence should have diversity 0.0
        assert diversity == 0.0
    
    def test_diversity_two_sentences_similar(self):
        """Test diversity calculation for two similar sentences."""
        generator = InsightGenerator()
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0]  # Similar to first
        ])
        
        diversity = generator._calculate_diversity(embeddings)
        
        # Should be low diversity (close to 0)
        assert 0.0 <= diversity <= 0.5
    
    def test_diversity_two_sentences_dissimilar(self):
        """Test diversity calculation for two dissimilar sentences."""
        generator = InsightGenerator()
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]  # Orthogonal to first
        ])
        
        diversity = generator._calculate_diversity(embeddings)
        
        # Should be high diversity (close to 1 for cosine distance)
        # Cosine distance between orthogonal vectors is 1.0
        # Average pairwise distance = 1.0
        assert 0.5 <= diversity <= 1.0
    
    def test_diversity_multiple_sentences(self):
        """Test diversity calculation for multiple sentences."""
        generator = InsightGenerator()
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        diversity = generator._calculate_diversity(embeddings)
        
        # All vectors are orthogonal, pairwise cosine distances = 1.0
        # Average = 1.0, but clipped to 1.0
        assert diversity == 1.0
    
    def test_diversity_clipped_range(self):
        """Test that diversity score is clipped to [0, 1] range."""
        generator = InsightGenerator()
        
        # Create embeddings that would give negative cosine distance
        # (shouldn't happen with normalized embeddings, but test clipping)
        with patch('sklearn.metrics.pairwise.cosine_distances') as mock_cosine:
            mock_cosine.return_value = np.array([
                [0.0, -0.5, 2.0],
                [-0.5, 0.0, 1.5],
                [2.0, 1.5, 0.0]
            ])
            
            embeddings = np.random.randn(3, 5)
            diversity = generator._calculate_diversity(embeddings)
            
            # Should be clipped to [0, 1]
            assert 0.0 <= diversity <= 1.0
    
    def test_diversity_deterministic(self):
        """Test that diversity calculation is deterministic."""
        generator = InsightGenerator()
        embeddings = np.random.RandomState(42).randn(10, 5)
        
        # Run twice
        div1 = generator._calculate_diversity(embeddings)
        div2 = generator._calculate_diversity(embeddings)
        
        # Should be identical
        assert div1 == div2
    
    def test_diversity_empty_embeddings(self):
        """Test diversity calculation with empty embeddings."""
        generator = InsightGenerator()
        embeddings = np.array([]).reshape(0, 5)
        
        diversity = generator._calculate_diversity(embeddings)
        
        # Empty embeddings should return 0.0
        assert diversity == 0.0
    
    def test_diversity_identical_embeddings(self):
        """Test diversity calculation with identical embeddings."""
        generator = InsightGenerator()
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ])
        
        diversity = generator._calculate_diversity(embeddings)
        
        # Identical embeddings should have diversity 0.0
        assert diversity == 0.0
    
    def test_diversity_normalized_embeddings(self):
        """Test diversity calculation with normalized embeddings."""
        generator = InsightGenerator()
        
        # Create normalized embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        diversity = generator._calculate_diversity(embeddings)
        
        # Cosine distance between orthogonal unit vectors is 1.0
        # Average of 3 choose 2 = 3 pairs, all distance 1.0
        # Average = 1.0, clipped to 1.0
        assert diversity == 1.0
    
    def test_diversity_with_mock_cosine(self):
        """Test diversity calculation with mocked cosine distances."""
        generator = InsightGenerator()
        
        with patch('sklearn.metrics.pairwise.cosine_distances') as mock_cosine:
            # Mock cosine distances
            mock_distances = np.array([
                [0.0, 0.3, 0.7],
                [0.3, 0.0, 0.5],
                [0.7, 0.5, 0.0]
            ])
            mock_cosine.return_value = mock_distances
            
            embeddings = np.random.randn(3, 5)
            diversity = generator._calculate_diversity(embeddings)
            
            # Average of upper triangle (excluding diagonal)
            # (0.3 + 0.7 + 0.5) / 3 = 0.5
            expected_diversity = (0.3 + 0.7 + 0.5) / 3
            
            assert abs(diversity - expected_diversity) < 1e-10