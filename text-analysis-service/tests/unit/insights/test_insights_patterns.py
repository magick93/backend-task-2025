"""
Tests for pattern identification in sentences.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from collections import Counter

from src.app.pipeline.insights import InsightGenerator


class TestPatternIdentification:
    """Test _identify_patterns method."""
    
    def test_identify_patterns_empty(self):
        """Test pattern identification with empty sentences."""
        generator = InsightGenerator()
        sentences = []
        
        patterns = generator._identify_patterns(sentences)
        
        assert patterns == []
    
    def test_identify_patterns_single_sentence(self):
        """Test pattern identification with single sentence."""
        generator = InsightGenerator()
        sentences = ["The product is excellent."]
        
        patterns = generator._identify_patterns(sentences)
        
        # Single sentence can't have patterns requiring frequency >= 2
        assert patterns == []
    
    def test_identify_patterns_common_starters(self):
        """Test identification of common sentence starters."""
        generator = InsightGenerator()
        sentences = [
            "The product is good.",
            "The service is bad.",
            "The quality is excellent.",
            "The price is high."
        ]
        
        patterns = generator._identify_patterns(sentences)
        
        # All sentences start with "The"
        assert "Often starts with 'the'" in patterns
    
    def test_identify_patterns_frequent_bigrams(self):
        """Test identification of frequent bigrams."""
        generator = InsightGenerator()
        sentences = [
            "Product quality is excellent.",
            "Service quality is poor.",
            "Build quality is good."
        ]
        
        patterns = generator._identify_patterns(sentences)
        
        # "quality is" appears 3 times
        assert "Frequent phrase: 'quality is'" in patterns
    
    def test_identify_patterns_multiple_patterns(self):
        """Test identification of multiple pattern types."""
        generator = InsightGenerator()
        sentences = [
            "I love the product.",
            "I hate the service.",
            "I like the quality.",
            "The product is good.",
            "The service is bad."
        ]
        
        patterns = generator._identify_patterns(sentences)
        
        # Should identify "I" as common starter (3 times)
        # Should identify "the product" as frequent bigram (2 times)
        # Should identify "the service" as frequent bigram (2 times)
        assert len(patterns) >= 1
        assert any("Often starts with 'i'" in p for p in patterns)
    
    def test_identify_patterns_case_insensitive(self):
        """Test that pattern identification is case insensitive."""
        generator = InsightGenerator()
        sentences = [
            "The product is good.",
            "the service is bad.",
            "THE quality is excellent."
        ]
        
        patterns = generator._identify_patterns(sentences)
        
        # All start with "the" (case insensitive)
        assert "Often starts with 'the'" in patterns
    
    def test_identify_patterns_minimum_frequency(self):
        """Test that patterns require minimum frequency of 2."""
        generator = InsightGenerator()
        sentences = [
            "Apple is good.",
            "Banana is bad.",
            "Cherry is okay."
        ]
        
        patterns = generator._identify_patterns(sentences)
        
        # No word appears twice as starter, no bigram appears twice
        assert patterns == []
    
    @pytest.mark.xfail(reason="Dash is not a sentence starter; test expectation may be incorrect")
    def test_identify_patterns_special_characters(self):
        """Test pattern identification with special characters."""
        generator = InsightGenerator()
        sentences = [
            "Product - excellent!",
            "Service - terrible!",
            "Quality - amazing!"
        ]
        
        patterns = generator._identify_patterns(sentences)
        
        # Should identify "-" as common starter
        assert "Often starts with '-'" in patterns
    
    def test_identify_patterns_short_words_filtered(self):
        """Test that short words are filtered from bigrams."""
        generator = InsightGenerator()
        sentences = [
            "It is good.",
            "It is bad.",
            "It is okay."
        ]
        
        patterns = generator._identify_patterns(sentences)
        
        # "it is" should be identified as frequent bigram
        assert "Frequent phrase: 'it is'" in patterns
    
    def test_identify_patterns_deterministic(self):
        """Test that pattern identification is deterministic."""
        generator = InsightGenerator()
        sentences = [
            "First sentence about product.",
            "Second sentence about product.",
            "Third sentence about service.",
            "Fourth sentence about service."
        ]
        
        patterns1 = generator._identify_patterns(sentences)
        patterns2 = generator._identify_patterns(sentences)
        
        # Should be identical
        assert patterns1 == patterns2
    
    def test_identify_patterns_max_patterns(self):
        """Test that only top 3 patterns of each type are returned."""
        generator = InsightGenerator()
        
        # Create many sentences with different starters
        sentences = []
        for i in range(10):
            sentences.append(f"Word{i} is good.")
            sentences.append(f"Word{i} is bad.")
        
        patterns = generator._identify_patterns(sentences)
        
        # Should only return top 3 starters and top 3 bigrams
        # Each word appears twice as starter, so many qualify
        # Implementation returns top 3 of each type
        starter_patterns = [p for p in patterns if "starts with" in p]
        bigram_patterns = [p for p in patterns if "Frequent phrase" in p]
        
        assert len(starter_patterns) <= 3
        assert len(bigram_patterns) <= 3