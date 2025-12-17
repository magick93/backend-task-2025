"""
Tests for InsightGenerator initialization and basic configuration.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.app.pipeline.insights import (
    InsightGenerator,
    get_insight_generator
)


class TestInsightGeneratorInitialization:
    """Test InsightGenerator initialization and configuration."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        generator = InsightGenerator()
        
        assert generator.max_title_length == 100
        assert generator.max_insights_per_cluster == 3
        assert generator.min_sentence_length == 10
        assert 'insight_templates' in generator.__dict__
        assert 'size' in generator.insight_templates
        assert 'sentiment_positive' in generator.insight_templates
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        generator = InsightGenerator(
            max_title_length=50,
            max_insights_per_cluster=5,
            min_sentence_length=5
        )
        
        assert generator.max_title_length == 50
        assert generator.max_insights_per_cluster == 5
        assert generator.min_sentence_length == 5
    
    def test_insight_templates_structure(self):
        """Test that insight templates have expected structure."""
        generator = InsightGenerator()
        templates = generator.insight_templates
        
        # Check required template categories
        required_categories = ['size', 'sentiment_positive', 'sentiment_negative', 
                              'sentiment_neutral', 'diversity']
        for category in required_categories:
            assert category in templates
            assert isinstance(templates[category], list)
            assert len(templates[category]) > 0
        
        # Check template strings contain placeholders where needed
        for template in templates['size']:
            assert '{count}' in template
        
        for template in templates['sentiment_positive']:
            assert '{confidence:' in template or '{confidence' in template
        
        for template in templates['sentiment_negative']:
            assert '{confidence:' in template or '{confidence' in template


class TestKeyTermExtraction:
    """Test _extract_key_terms method."""
    
    def test_extract_key_terms_basic(self):
        """Test basic key term extraction."""
        generator = InsightGenerator()
        
        # Mock preprocessor (not used in current implementation)
        mock_preprocessor = MagicMock()
        
        text = "The product is excellent and amazing. The service is excellent too."
        key_terms = generator._extract_key_terms(text, mock_preprocessor)
        
        # Should extract words with frequency >= 2 and length >= 3
        # "excellent" appears twice, "product", "service", "amazing" appear once
        # Only "excellent" should be included (frequency >= 2)
        assert 'excellent' in key_terms
        # Common words like 'the', 'and', 'too' should be filtered
        assert 'the' not in key_terms
        assert 'and' not in key_terms
    
    def test_extract_key_terms_no_repeats(self):
        """Test key term extraction when no words repeat."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        
        text = "This is a unique sentence with distinct words."
        key_terms = generator._extract_key_terms(text, mock_preprocessor)
        
        # No word appears twice, so should return empty list
        # (filtered_counts requires count >= 2)
        assert len(key_terms) == 0
    
    def test_extract_key_terms_common_words_filtered(self):
        """Test that common stop words are filtered out."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        
        # Create text where common words repeat but should be filtered
        text = "the the the and and for for with with this this"
        key_terms = generator._extract_key_terms(text, mock_preprocessor)
        
        # Common words should be filtered out
        assert len(key_terms) == 0
    
    def test_extract_key_terms_max_terms(self):
        """Test that extraction returns at most 10 terms."""
        generator = InsightGenerator()
        mock_preprocessor = MagicMock()
        
        # Create text with many repeating words
        words = [f"word{i}" for i in range(15)]
        text = " ".join(words * 3)  # Each word appears 3 times
        
        key_terms = generator._extract_key_terms(text, mock_preprocessor)
        
        # Should return at most 10 terms
        assert len(key_terms) <= 10
        # Should be sorted by frequency (descending)
        # Since all appear 3 times, order may vary but should contain some


class TestConvenienceFunctions:
    """Test convenience functions like get_insight_generator."""
    
    def test_get_insight_generator_default(self):
        """Test get_insight_generator with default parameters."""
        generator = get_insight_generator()
        
        assert isinstance(generator, InsightGenerator)
        assert generator.max_title_length == 100
        assert generator.max_insights_per_cluster == 3
    
    def test_get_insight_generator_custom(self):
        """Test get_insight_generator with custom parameters."""
        generator = get_insight_generator(
            max_title_length=50,
            max_insights_per_cluster=5,
            min_sentence_length=5
        )
        
        assert isinstance(generator, InsightGenerator)
        assert generator.max_title_length == 50
        assert generator.max_insights_per_cluster == 5
        assert generator.min_sentence_length == 5
    
    def test_get_insight_generator_singleton_behavior(self):
        """Test that get_insight_generator returns same instance with same args."""
        generator1 = get_insight_generator(max_title_length=60)
        generator2 = get_insight_generator(max_title_length=60)
        
        # Should be the same instance (singleton pattern)
        assert generator1 is generator2
        
        # Different args should create different instance
        generator3 = get_insight_generator(max_title_length=70)
        assert generator1 is not generator3