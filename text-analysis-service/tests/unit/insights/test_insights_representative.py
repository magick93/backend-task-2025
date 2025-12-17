"""
Tests for representative sentence selection.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.app.pipeline.insights import InsightGenerator


class TestRepresentativeSentenceSelection:
    """Test representative sentence selection logic."""
    
    def test_select_representative_sentences_basic(self):
        """Test basic representative sentence selection."""
        generator = InsightGenerator()
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        representative = generator._select_representative_sentences(sentences, embeddings)
        
        # Should return 3 sentences (default)
        assert len(representative) == 3
        # Should include all sentences when cluster size <= 3
        assert set(representative) == set(sentences)
    
    def test_select_representative_sentences_large_cluster(self):
        """Test representative selection from larger cluster."""
        generator = InsightGenerator()
        sentences = [f"Sentence {i}" for i in range(10)]
        # Create embeddings with clear centroid
        embeddings = np.array([
            [1.0, 0.0] if i < 5 else [0.0, 1.0] for i in range(10)
        ])
        
        representative = generator._select_representative_sentences(sentences, embeddings)
        
        # Should return 3 sentences (default)
        assert len(representative) == 3
        # All should be from the sentences list
        for rep in representative:
            assert rep in sentences
    
    def test_select_representative_sentences_custom_count(self):
        """Test representative selection with custom count."""
        generator = InsightGenerator(max_insights_per_cluster=5)
        sentences = [f"Sentence {i}" for i in range(10)]
        embeddings = np.random.randn(10, 5)
        
        representative = generator._select_representative_sentences(sentences, embeddings)
        
        # Should return 5 sentences (max_insights_per_cluster)
        assert len(representative) == 5
    
    def test_select_representative_sentences_single_sentence(self):
        """Test representative selection with single sentence."""
        generator = InsightGenerator()
        sentences = ["Only one sentence."]
        embeddings = np.array([[1.0, 2.0, 3.0]])
        
        representative = generator._select_representative_sentences(sentences, embeddings)
        
        # Should return the single sentence
        assert representative == ["Only one sentence."]
    
    def test_select_representative_sentences_empty(self):
        """Test representative selection with empty cluster."""
        generator = InsightGenerator()
        sentences = []
        embeddings = np.array([]).reshape(0, 5)
        
        representative = generator._select_representative_sentences(sentences, embeddings)
        
        # Should return empty list
        assert representative == []
    
    def test_representative_sentences_deterministic(self):
        """Test that representative selection is deterministic."""
        generator = InsightGenerator()
        sentences = [f"Sentence {i}" for i in range(10)]
        embeddings = np.random.RandomState(42).randn(10, 5)
        
        # Run twice with same inputs
        rep1 = generator._select_representative_sentences(sentences, embeddings)
        rep2 = generator._select_representative_sentences(sentences, embeddings)
        
        # Should be identical
        assert rep1 == rep2
    
    def test_representative_sentences_closest_to_centroid(self):
        """Test that selected sentences are closest to centroid."""
        generator = InsightGenerator()
        
        # Create embeddings where first 3 are close to each other (near centroid)
        # and others are far away
        sentences = [f"Sentence {i}" for i in range(6)]
        embeddings = np.array([
            [1.0, 1.0],    # Close to centroid (1,1)
            [1.1, 0.9],    # Close to centroid
            [0.9, 1.1],    # Close to centroid
            [5.0, 5.0],    # Far from centroid
            [-3.0, -3.0],  # Far from centroid
            [10.0, 0.0]    # Far from centroid
        ])
        
        representative = generator._select_representative_sentences(sentences, embeddings)
        
        # Centroid is approximately (1,1)
        # First 3 sentences should be selected
        assert "Sentence 0" in representative
        assert "Sentence 1" in representative
        assert "Sentence 2" in representative
    
    def test_representative_sentences_with_duplicate_embeddings(self):
        """Test representative selection with duplicate embeddings."""
        generator = InsightGenerator()
        sentences = ["A", "B", "C", "D", "E"]
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0],  # Duplicate of first
            [0.0, 1.0],
            [0.0, 1.0],  # Duplicate of third
            [0.5, 0.5]
        ])
        
        representative = generator._select_representative_sentences(sentences, embeddings)
        
        # Should still return 3 sentences
        assert len(representative) == 3
        # All should be unique
        assert len(set(representative)) == len(representative)