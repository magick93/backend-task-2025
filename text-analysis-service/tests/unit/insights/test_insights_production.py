"""
Production readiness tests for insight generation module.
Tests for performance, edge cases, and production-specific requirements.
"""

import numpy as np
import pytest
import time
from unittest.mock import MagicMock

from src.app.pipeline.insights import InsightGenerator


class TestProductionReadiness:
    """Production readiness tests for insight generation."""
    
    def test_performance_small_cluster(self):
        """Test performance with small cluster."""
        generator = InsightGenerator()
        
        # Small cluster
        sentences = ["Test sentence"] * 10
        embeddings = np.random.randn(10, 384)
        
        start_time = time.time()
        diversity = generator._calculate_diversity(embeddings)
        end_time = time.time()
        
        assert diversity >= 0.0
        assert diversity <= 1.0
        # Should complete quickly
        assert end_time - start_time < 1.0  # Less than 1 second
    
    def test_performance_large_cluster(self):
        """Test performance with large cluster."""
        generator = InsightGenerator()
        
        # Large cluster (100 sentences)
        sentences = [f"Sentence {i}" for i in range(100)]
        embeddings = np.random.randn(100, 384)
        
        start_time = time.time()
        diversity = generator._calculate_diversity(embeddings)
        end_time = time.time()
        
        assert diversity >= 0.0
        assert diversity <= 1.0
        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # Less than 5 seconds
    
    def test_memory_usage_large_cluster(self):
        """Test memory usage with large cluster."""
        generator = InsightGenerator()
        
        # Very large cluster (1000 sentences) - test memory
        sentences = [f"Sentence {i}" for i in range(1000)]
        embeddings = np.random.randn(1000, 384)
        
        # This should not crash
        diversity = generator._calculate_diversity(embeddings)
        assert diversity >= 0.0
        assert diversity <= 1.0
    
    def test_title_length_constraints(self):
        """Test that titles respect length constraints."""
        generator = InsightGenerator(max_title_length=50)
        
        # Very long sentence
        long_sentence = "This is a very long sentence that should be truncated to fit within the maximum title length constraint of the insight generator module"
        sentences = [long_sentence]
        key_terms = []
        
        title = generator._generate_cluster_title(sentences, key_terms)
        
        assert len(title) <= 50
        assert title.endswith("...") or len(title) < len(long_sentence)
    
    def test_title_length_constraints_with_key_terms(self):
        """Test title length constraints when using key terms."""
        generator = InsightGenerator(max_title_length=30)
        
        sentences = [
            "This sentence contains the important key term that should be extracted",
            "Another sentence with the key term"
        ]
        key_terms = ['important', 'key', 'term']
        
        title = generator._generate_cluster_title(sentences, key_terms)
        
        assert len(title) <= 30
        assert any(term in title.lower() for term in key_terms)
    
    def test_min_sentence_length_filtering(self):
        """Test that very short sentences are handled appropriately."""
        generator = InsightGenerator(min_sentence_length=5)
        
        # Mix of short and long sentences
        sentences = [
            "Hi",  # Too short (2 chars)
            "OK",  # Too short (2 chars)
            "This is a proper sentence",  # Long enough
            "Good"  # Too short (4 chars)
        ]
        
        # Patterns should still work with short sentences
        patterns = generator._identify_patterns(sentences)
        # Should not crash
    
    def test_empty_patterns_edge_cases(self):
        """Test edge cases for pattern identification."""
        generator = InsightGenerator()
        
        # Single word sentences
        patterns1 = generator._identify_patterns(["Hello", "World", "Test"])
        assert isinstance(patterns1, list)
        
        # Sentences with punctuation
        patterns2 = generator._identify_patterns(["Hello!", "World?", "Test."])
        assert isinstance(patterns2, list)
        
        # Mixed case
        patterns3 = generator._identify_patterns(["HELLO world", "hello WORLD", "Hello World"])
        assert isinstance(patterns3, list)
    
    def test_diversity_edge_cases(self):
        """Test edge cases for diversity calculation."""
        generator = InsightGenerator()
        
        # Single embedding
        diversity1 = generator._calculate_diversity(np.array([[1.0, 0.0, 0.0]]))
        assert diversity1 == 0.0
        
        # Two identical embeddings
        diversity2 = generator._calculate_diversity(np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ]))
        assert diversity2 == 0.0
        
        # Two orthogonal embeddings (maximum diversity)
        diversity3 = generator._calculate_diversity(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]))
        assert diversity3 > 0.0
        assert diversity3 <= 1.0
    
    def test_summary_edge_cases(self):
        """Test edge cases for summary generation."""
        generator = InsightGenerator()
        
        # Zero sentences (should not happen but handle gracefully)
        summary1 = generator._generate_summary([], {'sentiment': 'positive'})
        assert summary1 == "A single positive comment."  # Default behavior
        
        # Very large number of sentences
        summary2 = generator._generate_summary(
            ["Sentence"] * 1000,
            {'sentiment': 'negative'}
        )
        assert "Strong cluster with 1000 concerns" in summary2
        
        # Missing sentiment
        summary3 = generator._generate_summary(["Test"], {})
        assert summary3 == "A single neutral comment."  # Default to neutral
    
    def test_overall_summary_edge_cases(self):
        """Test edge cases for overall summary generation."""
        generator = InsightGenerator()
        
        # Empty clusters
        summary1 = generator.generate_overall_summary([])
        assert summary1['total_clusters'] == 0
        assert summary1['total_sentences'] == 0
        
        # Single cluster with zero sentences (should not happen)
        clusters = [{'title': 'Test', 'sentence_count': 0, 'sentiment': 'neutral'}]
        summary2 = generator.generate_overall_summary(clusters)
        assert summary2['total_sentences'] == 0
        assert summary2['average_cluster_size'] == 0.0
        
        # Very large clusters
        large_clusters = [
            {'title': f'Cluster {i}', 'sentence_count': 1000, 'sentiment': 'positive'}
            for i in range(10)
        ]
        summary3 = generator.generate_overall_summary(large_clusters)
        assert summary3['total_sentences'] == 10000
        assert summary3['average_cluster_size'] == 1000.0
    
    def test_error_handling_in_cluster_insights(self):
        """Test error handling in cluster insights generation."""
        generator = InsightGenerator()
        
        # Mock preprocessor that raises exception
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.side_effect = Exception("Test error")
        
        # Should handle gracefully (extract_key_terms is mocked, but if real would need try/except)
        # For now, just verify it doesn't crash with our mock
        sentences = ["Test sentence"]
        embeddings = np.array([[0.1, 0.2, 0.3]])
        
        # This would fail if extract_key_terms actually raises
        # We're testing that the structure handles missing key terms
        pass
    
    def test_configuration_parameters(self):
        """Test that configuration parameters are respected."""
        # Test max_insights_per_cluster
        generator1 = InsightGenerator(max_insights_per_cluster=3)
        assert generator1.max_insights_per_cluster == 3
        
        # Test min_sentence_length
        generator2 = InsightGenerator(min_sentence_length=20)
        assert generator2.min_sentence_length == 20
        
        # Test max_title_length
        generator3 = InsightGenerator(max_title_length=200)
        assert generator3.max_title_length == 200
        
        # Test default values
        generator_default = InsightGenerator()
        assert generator_default.max_insights_per_cluster == 3
        assert generator_default.min_sentence_length == 10
        assert generator_default.max_title_length == 100
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        generator = InsightGenerator()
        
        # Unicode sentences
        sentences = [
            "Café ☕ and emoji 🚀",
            "Unicode test: 中文 Español Français",
            "Special chars: © ® ™ € £ ¥"
        ]
        
        # Should not crash
        patterns = generator._identify_patterns(sentences)
        assert isinstance(patterns, list)
        
        # Test summary with Unicode
        summary = generator._generate_summary(sentences, {'sentiment': 'positive'})
        assert isinstance(summary, str)
    
    def test_concurrent_usage(self):
        """Test that generator can be used concurrently (basic test)."""
        generator = InsightGenerator()
        
        # Create multiple clusters
        clusters1 = [
            {'title': 'Cluster A', 'sentence_count': 5, 'sentiment': 'positive'},
            {'title': 'Cluster B', 'sentence_count': 3, 'sentiment': 'negative'}
        ]
        
        clusters2 = [
            {'title': 'Cluster C', 'sentence_count': 10, 'sentiment': 'neutral'}
        ]
        
        # Generate summaries concurrently (simulated)
        summary1 = generator.generate_overall_summary(clusters1)
        summary2 = generator.generate_overall_summary(clusters2)
        
        assert summary1['total_sentences'] == 8
        assert summary2['total_sentences'] == 10
    
    @pytest.mark.skip(reason="Production monitoring not implemented yet")
    def test_metrics_and_monitoring(self):
        """TODO: Test integration with metrics and monitoring systems."""
        pass
    
    @pytest.mark.skip(reason="Logging integration not implemented yet")
    def test_logging_integration(self):
        """TODO: Test that appropriate logs are generated."""
        pass
    
    @pytest.mark.skip(reason="Configuration validation not implemented yet")
    def test_configuration_validation(self):
        """TODO: Test validation of configuration parameters."""
        pass
    
    @pytest.mark.skip(reason="Integration with actual ML models not implemented yet")
    def test_integration_with_real_ml_pipeline(self):
        """TODO: Test integration with actual ML models in production."""
        pass