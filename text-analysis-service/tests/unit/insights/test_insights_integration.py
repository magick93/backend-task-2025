"""
Integration tests for insight generation module.
Tests the full generate_cluster_insights method and integration with other components.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.app.pipeline.insights import InsightGenerator


class TestClusterInsightsIntegration:
    """Integration tests for cluster insights generation."""
    
    def test_generate_cluster_insights_basic(self):
        """Test basic cluster insights generation."""
        generator = InsightGenerator()
        
        # Mock preprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = ['product', 'quality', 'service']
        
        # Test data
        cluster_id = 1
        sentences = [
            "The product quality is excellent",
            "Service was great and product worked well",
            "Good quality product with excellent service"
        ]
        embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        sentiment_info = {
            'sentiment': 'positive',
            'confidence': 0.85
        }
        
        # Generate insights
        insights = generator.generate_cluster_insights(
            cluster_id=cluster_id,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=mock_preprocessor
        )
        
        # Verify structure
        assert insights is not None
        assert insights['cluster_id'] == cluster_id
        assert insights['title'] is not None
        assert len(insights['representative_sentences']) == 3
        assert insights['sentiment'] == 'positive'
        assert insights['sentiment_confidence'] == 0.85
        assert insights['key_terms'] == ['product', 'quality', 'service']
        assert insights['sentence_count'] == 3
        assert 0.0 <= insights['diversity_score'] <= 1.0
        assert isinstance(insights['patterns'], list)
        assert insights['summary'] is not None
        
        # Verify preprocessor was called
        mock_preprocessor.extract_key_terms.assert_called_once()
    
    def test_generate_cluster_insights_empty_sentences(self):
        """Test cluster insights with empty sentences list."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        
        insights = generator.generate_cluster_insights(
            cluster_id=1,
            sentences=[],
            embeddings=np.array([]),
            sentiment_info={'sentiment': 'neutral'},
            preprocessor=mock_preprocessor
        )
        
        assert insights is None
        mock_preprocessor.extract_key_terms.assert_not_called()
    
    def test_generate_cluster_insights_single_sentence(self):
        """Test cluster insights with single sentence."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = ['test']
        
        sentences = ["Single test sentence"]
        embeddings = np.array([[0.1, 0.2, 0.3]])
        
        insights = generator.generate_cluster_insights(
            cluster_id=2,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'neutral', 'confidence': 0.5},
            preprocessor=mock_preprocessor
        )
        
        assert insights is not None
        assert insights['sentence_count'] == 1
        assert insights['diversity_score'] == 0.0  # Single sentence has zero diversity
        assert len(insights['representative_sentences']) == 1
        assert insights['representative_sentences'][0] == sentences[0]
    
    def test_generate_cluster_insights_title_generation(self):
        """Test that title generation works correctly."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = ['key', 'term']
        
        sentences = [
            "This is a sentence with key term in it",
            "Another sentence without the term",
            "Third sentence also has key term"
        ]
        embeddings = np.random.randn(3, 384)
        
        insights = generator.generate_cluster_insights(
            cluster_id=3,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'positive', 'confidence': 0.9},
            preprocessor=mock_preprocessor
        )
        
        assert insights['title'] is not None
        assert len(insights['title']) <= generator.max_title_length
        
        # Title should contain key term or be based on first sentence
        title_lower = insights['title'].lower()
        assert 'key' in title_lower or 'term' in title_lower or 'sentence' in title_lower
    
    def test_generate_cluster_insights_representative_sentences(self):
        """Test that representative sentences are selected correctly."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = []
        
        # Create embeddings with clear centroid
        sentences = ["A", "B", "C", "D", "E"]
        # Make first three sentences close to each other, last two farther
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Close to centroid
            [0.9, 0.1, 0.0],  # Close to centroid
            [0.8, 0.2, 0.0],  # Close to centroid
            [0.0, 1.0, 0.0],  # Far from centroid
            [0.0, 0.0, 1.0]   # Far from centroid
        ])
        
        insights = generator.generate_cluster_insights(
            cluster_id=4,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'neutral', 'confidence': 0.5},
            preprocessor=mock_preprocessor
        )
        
        # Should return 3 representative sentences (closest to centroid)
        assert len(insights['representative_sentences']) == 3
        # First three sentences should be representative (closest to centroid)
        assert all(s in sentences[:3] for s in insights['representative_sentences'])
    
    def test_generate_cluster_insights_diversity_calculation(self):
        """Test diversity score calculation."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = []
        
        # Test with identical embeddings (zero diversity)
        sentences = ["A", "B", "C"]
        identical_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        
        insights1 = generator.generate_cluster_insights(
            cluster_id=5,
            sentences=sentences,
            embeddings=identical_embeddings,
            sentiment_info={'sentiment': 'neutral', 'confidence': 0.5},
            preprocessor=mock_preprocessor
        )
        
        assert insights1['diversity_score'] == 0.0
        
        # Test with diverse embeddings (higher diversity)
        diverse_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        insights2 = generator.generate_cluster_insights(
            cluster_id=6,
            sentences=sentences,
            embeddings=diverse_embeddings,
            sentiment_info={'sentiment': 'neutral', 'confidence': 0.5},
            preprocessor=mock_preprocessor
        )
        
        assert insights2['diversity_score'] > 0.0
        assert insights2['diversity_score'] <= 1.0
    
    def test_generate_cluster_insights_pattern_identification(self):
        """Test pattern identification in sentences."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = []
        
        # Sentences with common patterns
        sentences = [
            "The product is great",
            "The service is excellent",
            "The quality is outstanding",
            "I love the product",
            "Service could be better"  # Different starter
        ]
        embeddings = np.random.randn(5, 384)
        
        insights = generator.generate_cluster_insights(
            cluster_id=7,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'positive', 'confidence': 0.8},
            preprocessor=mock_preprocessor
        )
        
        # Should identify "The" as common starter
        patterns = insights['patterns']
        assert any("often starts with 'the'" in p.lower() for p in patterns)
    
    def test_generate_cluster_insights_with_missing_sentiment_info(self):
        """Test cluster insights with incomplete sentiment info."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = ['test']
        
        sentences = ["Test sentence"]
        embeddings = np.array([[0.1, 0.2, 0.3]])
        
        # Missing confidence
        insights1 = generator.generate_cluster_insights(
            cluster_id=8,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'positive'},  # No confidence
            preprocessor=mock_preprocessor
        )
        
        assert insights1['sentiment'] == 'positive'
        assert insights1['sentiment_confidence'] == 0.0  # Default
        
        # Missing sentiment entirely
        insights2 = generator.generate_cluster_insights(
            cluster_id=9,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={},  # Empty
            preprocessor=mock_preprocessor
        )
        
        assert insights2['sentiment'] == 'neutral'  # Default
        assert insights2['sentiment_confidence'] == 0.0  # Default
    
    def test_generate_cluster_insights_deterministic(self):
        """Test that cluster insights generation is deterministic."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = ['deterministic', 'test']
        
        sentences = ["Deterministic test sentence one", "Deterministic test sentence two"]
        embeddings = np.random.randn(2, 384)
        sentiment_info = {'sentiment': 'positive', 'confidence': 0.9}
        
        # Generate insights twice
        insights1 = generator.generate_cluster_insights(
            cluster_id=10,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=mock_preprocessor
        )
        
        insights2 = generator.generate_cluster_insights(
            cluster_id=10,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=mock_preprocessor
        )
        
        # Should be identical
        assert insights1 == insights2
    
    def test_generate_cluster_insights_edge_cases(self):
        """Test edge cases in cluster insights generation."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        mock_preprocessor.extract_key_terms.return_value = []
        
        # Very long sentences
        long_sentence = "word " * 100
        sentences = [long_sentence]
        embeddings = np.array([[0.1] * 384])
        
        insights = generator.generate_cluster_insights(
            cluster_id=11,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'neutral', 'confidence': 0.5},
            preprocessor=mock_preprocessor
        )
        
        assert insights is not None
        assert len(insights['title']) <= generator.max_title_length
        
        # Very short sentences
        short_sentences = ["a", "b", "c"]
        embeddings = np.random.randn(3, 384)
        
        insights = generator.generate_cluster_insights(
            cluster_id=12,
            sentences=short_sentences,
            embeddings=embeddings,
            sentiment_info={'sentiment': 'positive', 'confidence': 0.7},
            preprocessor=mock_preprocessor
        )
        
        assert insights is not None
        assert insights['sentence_count'] == 3
    
    @pytest.mark.skip(reason="Integration with actual preprocessor not implemented yet")
    def test_integration_with_real_preprocessor(self):
        """TODO: Test integration with actual TextPreprocessor."""
        pass
    
    @pytest.mark.skip(reason="Performance testing not implemented yet")
    def test_performance_large_cluster(self):
        """TODO: Test performance with large clusters."""
        pass