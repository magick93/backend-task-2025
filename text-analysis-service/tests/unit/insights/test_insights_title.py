"""
Tests for cluster title generation.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.app.pipeline.insights import InsightGenerator


class TestClusterTitleGeneration:
    """Test _generate_cluster_title method."""
    
    def test_title_from_key_terms(self):
        """Test title generation using key terms."""
        generator = InsightGenerator(max_title_length=100)
        sentences = ["Sentence one.", "Sentence two."]
        key_terms = ["product", "quality", "improvement", "feedback"]
        
        title = generator._generate_cluster_title(sentences, key_terms)
        
        # Should create title from first 3 key terms
        expected = "Product Quality Improvement"
        assert title == expected
    
    def test_title_truncation(self):
        """Test title truncation when too long."""
        generator = InsightGenerator(max_title_length=20)
        sentences = ["Sentence one.", "Sentence two."]
        key_terms = ["verylongproductname", "qualityimprovement", "feedbackanalysis"]
        
        title = generator._generate_cluster_title(sentences, key_terms)
        
        # Should be truncated to max_title_length
        assert len(title) <= 20
        assert title.endswith("...") or len(title) <= 20
    
    def test_title_fallback_to_first_sentence(self):
        """Test title generation falls back to first sentence when no key terms."""
        generator = InsightGenerator(max_title_length=50)
        sentences = ["This is the first sentence about the product.", "Second sentence."]
        key_terms = []  # No key terms
        
        title = generator._generate_cluster_title(sentences, key_terms)
        
        # Should use first sentence
        assert title == sentences[0]
    
    def test_title_fallback_truncation(self):
        """Test title fallback truncation when first sentence is too long."""
        generator = InsightGenerator(max_title_length=30)
        sentences = ["This is a very long sentence that exceeds the maximum title length allowed.", "Short."]
        key_terms = []
        
        title = generator._generate_cluster_title(sentences, key_terms)
        
        # Should be truncated
        assert len(title) <= 30
        assert title.endswith("...")
    
    def test_empty_cluster_title(self):
        """Test title generation for empty cluster."""
        generator = InsightGenerator()
        sentences = []
        key_terms = []
        
        title = generator._generate_cluster_title(sentences, key_terms)
        
        assert title == "Empty Cluster"