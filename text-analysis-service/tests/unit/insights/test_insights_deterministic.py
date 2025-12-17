"""
Deterministic behavior tests for insight generation module.
Tests that operations produce identical results given identical inputs.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.app.pipeline.insights import InsightGenerator


class TestDeterministicBehavior:
    """Test deterministic behavior of insight generation."""
    
    def test_title_generation_deterministic(self):
        """Test that title generation is deterministic."""
        generator = InsightGenerator()
        sentences = [
            "The product quality is excellent and service is great",
            "Service could be improved but product is good",
            "Overall good experience with the product"
        ]
        key_terms = ['product', 'quality', 'service']
        
        # Generate title multiple times
        title1 = generator._generate_cluster_title(sentences, key_terms)
        title2 = generator._generate_cluster_title(sentences, key_terms)
        title3 = generator._generate_cluster_title(sentences, key_terms)
        
        # All should be identical
        assert title1 == title2 == title3
    
    def test_title_generation_deterministic_with_different_order(self):
        """Test that title generation is deterministic even with sentence order changes."""
        generator = InsightGenerator()
        
        # Same sentences, different order
        sentences1 = ["A", "B", "C"]
        sentences2 = ["C", "A", "B"]
        key_terms = []
        
        title1 = generator._generate_cluster_title(sentences1, key_terms)
        title2 = generator._generate_cluster_title(sentences2, key_terms)
        
        # Should still be deterministic (uses first sentence containing key term or first sentence)
        # Since no key terms, uses first sentence of each list
        assert title1 == "A"
        assert title2 == "C"
        # Not equal because different first sentences, but each call with same input is deterministic
    
    def test_diversity_calculation_deterministic(self):
        """Test that diversity calculation is deterministic."""
        generator = InsightGenerator()
        
        # Create embeddings
        np.random.seed(42)
        embeddings = np.random.randn(5, 384)
        
        # Calculate diversity multiple times
        diversity1 = generator._calculate_diversity(embeddings)
        diversity2 = generator._calculate_diversity(embeddings)
        diversity3 = generator._calculate_diversity(embeddings)
        
        # All should be identical
        assert diversity1 == diversity2 == diversity3
        assert 0.0 <= diversity1 <= 1.0
    
    def test_diversity_calculation_deterministic_with_identical_embeddings(self):
        """Test diversity calculation with identical embeddings is deterministic."""
        generator = InsightGenerator()
        
        # Identical embeddings
        embeddings = np.ones((3, 384))
        
        diversity1 = generator._calculate_diversity(embeddings)
        diversity2 = generator._calculate_diversity(embeddings)
        
        assert diversity1 == diversity2 == 0.0
    
    def test_pattern_identification_deterministic(self):
        """Test that pattern identification is deterministic."""
        generator = InsightGenerator()
        
        sentences = [
            "The product is great",
            "The service is excellent",
            "The quality is outstanding",
            "I love the product",
            "Service could be better"
        ]
        
        # Identify patterns multiple times
        patterns1 = generator._identify_patterns(sentences)
        patterns2 = generator._identify_patterns(sentences)
        patterns3 = generator._identify_patterns(sentences)
        
        # All should be identical
        assert patterns1 == patterns2 == patterns3
    
    def test_summary_generation_deterministic(self):
        """Test that summary generation is deterministic."""
        generator = InsightGenerator()
        
        sentences = ["Test sentence"] * 7
        sentiment_info = {'sentiment': 'positive'}
        
        summary1 = generator._generate_summary(sentences, sentiment_info)
        summary2 = generator._generate_summary(sentences, sentiment_info)
        summary3 = generator._generate_summary(sentences, sentiment_info)
        
        assert summary1 == summary2 == summary3
    
    def test_overall_summary_generation_deterministic(self):
        """Test that overall summary generation is deterministic."""
        generator = InsightGenerator()
        
        clusters = [
            {
                'title': f'Cluster {i}',
                'sentence_count': i + 1,
                'sentiment': 'positive' if i % 2 == 0 else 'negative'
            }
            for i in range(5)
        ]
        
        summary1 = generator.generate_overall_summary(clusters)
        summary2 = generator.generate_overall_summary(clusters)
        summary3 = generator.generate_overall_summary(clusters)
        
        assert summary1 == summary2 == summary3
    
    def test_cluster_insights_deterministic(self):
        """Test that full cluster insights generation is deterministic."""
        generator = InsightGenerator()
        
        # Mock preprocessor with deterministic key terms
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = ['deterministic', 'test']
        
        # Create deterministic embeddings
        np.random.seed(123)
        sentences = ["Deterministic test sentence"] * 3
        embeddings = np.random.randn(3, 384)
        sentiment_info = {'sentiment': 'positive', 'confidence': 0.9}
        
        # Generate insights multiple times
        insights1 = generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=mock_preprocessor
        )
        
        insights2 = generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=mock_preprocessor
        )
        
        insights3 = generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=mock_preprocessor
        )
        
        # All should be identical
        assert insights1 == insights2 == insights3
    
    def test_representative_sentences_deterministic(self):
        """Test that representative sentence selection is deterministic."""
        generator = InsightGenerator()
        
        # Create embeddings with clear ordering
        sentences = ["A", "B", "C", "D", "E"]
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Closest to centroid (0.7, 0.2, 0.1)
            [0.8, 0.2, 0.0],  # Second closest
            [0.6, 0.4, 0.0],  # Third closest
            [0.0, 1.0, 0.0],  # Far
            [0.0, 0.0, 1.0]   # Far
        ])
        
        # Mock preprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = []
        
        # Generate insights multiple times
        insights1 = generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'neutral'},
            preprocessor=mock_preprocessor
        )
        
        insights2 = generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'neutral'},
            preprocessor=mock_preprocessor
        )
        
        # Representative sentences should be identical
        rep1 = insights1['representative_sentences']
        rep2 = insights2['representative_sentences']
        
        assert rep1 == rep2
        # Should be ["A", "B", "C"] in that order (closest to centroid)
        assert rep1 == ["A", "B", "C"]
    
    def test_deterministic_across_multiple_instances(self):
        """Test that different generator instances produce same results."""
        generator1 = InsightGenerator()
        generator2 = InsightGenerator()
        
        sentences = ["Test sentence"] * 5
        sentiment_info = {'sentiment': 'negative'}
        
        summary1 = generator1._generate_summary(sentences, sentiment_info)
        summary2 = generator2._generate_summary(sentences, sentiment_info)
        
        assert summary1 == summary2
    
    def test_deterministic_with_random_embeddings(self):
        """Test deterministic behavior with random embeddings (fixed seed)."""
        generator = InsightGenerator()
        
        # Fix random seed for reproducibility
        np.random.seed(999)
        embeddings1 = np.random.randn(4, 384)
        
        np.random.seed(999)  # Reset seed
        embeddings2 = np.random.randn(4, 384)
        
        # Should be identical
        assert np.array_equal(embeddings1, embeddings2)
        
        # Diversity calculations should be identical
        diversity1 = generator._calculate_diversity(embeddings1)
        diversity2 = generator._calculate_diversity(embeddings2)
        
        assert diversity1 == diversity2
    
    def test_edge_case_deterministic_empty_sentences(self):
        """Test deterministic behavior with edge cases."""
        generator = InsightGenerator()
        
        # Empty sentences list
        patterns1 = generator._identify_patterns([])
        patterns2 = generator._identify_patterns([])
        
        assert patterns1 == patterns2 == []
        
        # Single sentence
        patterns3 = generator._identify_patterns(["Single"])
        patterns4 = generator._identify_patterns(["Single"])
        
        assert patterns3 == patterns4
    
    @pytest.mark.skip(reason="Production-specific deterministic behavior not implemented yet")
    def test_deterministic_with_real_ml_models(self):
        """TODO: Test deterministic behavior with actual ML models (requires fixed seeds)."""
        pass
    
    @pytest.mark.skip(reason="Concurrency deterministic testing not implemented yet")
    def test_deterministic_under_concurrency(self):
        """TODO: Test that operations remain deterministic under concurrent execution."""
        pass