"""
Tests for summary generation in insight generation module.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.app.pipeline.insights import InsightGenerator


class TestSummaryGeneration:
    """Test summary generation functionality."""
    
    def test_generate_summary_single_positive(self):
        """Test summary generation for a single positive comment."""
        generator = InsightGenerator()
        sentences = ["Great product!"]
        sentiment_info = {'sentiment': 'positive'}
        
        summary = generator._generate_summary(sentences, sentiment_info)
        assert summary == "A single positive comment."
    
    def test_generate_summary_single_negative(self):
        """Test summary generation for a single negative comment."""
        generator = InsightGenerator()
        sentences = ["Terrible service"]
        sentiment_info = {'sentiment': 'negative'}
        
        summary = generator._generate_summary(sentences, sentiment_info)
        assert summary == "A single negative comment."
    
    def test_generate_summary_single_neutral(self):
        """Test summary generation for a single neutral comment."""
        generator = InsightGenerator()
        sentences = ["It's okay"]
        sentiment_info = {'sentiment': 'neutral'}
        
        summary = generator._generate_summary(sentences, sentiment_info)
        assert summary == "A single neutral comment."
    
    def test_generate_summary_few_positive(self):
        """Test summary generation for a few positive comments."""
        generator = InsightGenerator()
        sentences = ["Good", "Great", "Excellent"]
        sentiment_info = {'sentiment': 'positive'}
        
        summary = generator._generate_summary(sentences, sentiment_info)
        assert summary == "A few positive feedback on this topic."
    
    def test_generate_summary_few_negative(self):
        """Test summary generation for a few negative comments."""
        generator = InsightGenerator()
        sentences = ["Bad", "Terrible", "Awful"]
        sentiment_info = {'sentiment': 'negative'}
        
        summary = generator._generate_summary(sentences, sentiment_info)
        assert summary == "A few concerns on this topic."
    
    def test_generate_summary_multiple_positive(self):
        """Test summary generation for multiple positive comments."""
        generator = InsightGenerator()
        sentences = [f"Good {i}" for i in range(8)]  # 8 sentences
        sentiment_info = {'sentiment': 'positive'}
        
        summary = generator._generate_summary(sentences, sentiment_info)
        assert summary == "Multiple positive feedback from 8 respondents."
    
    def test_generate_summary_strong_cluster(self):
        """Test summary generation for a strong cluster."""
        generator = InsightGenerator()
        sentences = [f"Comment {i}" for i in range(15)]  # 15 sentences
        sentiment_info = {'sentiment': 'negative'}
        
        summary = generator._generate_summary(sentences, sentiment_info)
        assert summary == "Strong cluster with 15 concerns."
    
    def test_generate_summary_edge_cases(self):
        """Test summary generation with edge cases."""
        generator = InsightGenerator()
        
        # Empty sentences list
        summary = generator._generate_summary([], {'sentiment': 'positive'})
        assert summary == "A single positive comment."  # Default behavior
        
        # Missing sentiment info
        summary = generator._generate_summary(["Test"], {})
        assert summary == "A single neutral comment."  # Default to neutral
    
    def test_generate_overall_summary_empty(self):
        """Test overall summary generation with empty clusters."""
        generator = InsightGenerator()
        clusters = []
        
        summary = generator.generate_overall_summary(clusters)
        assert summary['total_clusters'] == 0
        assert summary['total_sentences'] == 0
    
    def test_generate_overall_summary_single_cluster(self):
        """Test overall summary generation with single cluster."""
        generator = InsightGenerator()
        clusters = [
            {
                'title': 'Test Cluster',
                'sentence_count': 5,
                'sentiment': 'positive'
            }
        ]
        
        summary = generator.generate_overall_summary(clusters)
        assert summary['total_clusters'] == 1
        assert summary['total_sentences'] == 5
        assert summary['sentiment_distribution'] == {'positive': 5}
        assert summary['sentiment_percentages'] == {'positive': 1.0}
        assert summary['dominant_sentiment'] == 'positive'
        assert summary['largest_cluster']['title'] == 'Test Cluster'
        assert summary['largest_cluster']['size'] == 5
        assert summary['largest_cluster']['sentiment'] == 'positive'
        assert summary['average_cluster_size'] == 5.0
    
    def test_generate_overall_summary_multiple_clusters(self):
        """Test overall summary generation with multiple clusters."""
        generator = InsightGenerator()
        clusters = [
            {
                'title': 'Positive Cluster',
                'sentence_count': 10,
                'sentiment': 'positive'
            },
            {
                'title': 'Negative Cluster',
                'sentence_count': 5,
                'sentiment': 'negative'
            },
            {
                'title': 'Neutral Cluster',
                'sentence_count': 3,
                'sentiment': 'neutral'
            }
        ]
        
        summary = generator.generate_overall_summary(clusters)
        assert summary['total_clusters'] == 3
        assert summary['total_sentences'] == 18
        assert summary['sentiment_distribution'] == {
            'positive': 10,
            'negative': 5,
            'neutral': 3
        }
        assert summary['sentiment_percentages']['positive'] == pytest.approx(10/18)
        assert summary['sentiment_percentages']['negative'] == pytest.approx(5/18)
        assert summary['sentiment_percentages']['neutral'] == pytest.approx(3/18)
        assert summary['dominant_sentiment'] == 'positive'
        assert summary['largest_cluster']['title'] == 'Positive Cluster'
        assert summary['largest_cluster']['size'] == 10
        assert summary['average_cluster_size'] == pytest.approx(18/3)
    
    def test_generate_overall_summary_tie_break(self):
        """Test overall summary with tied sentiment distribution."""
        generator = InsightGenerator()
        clusters = [
            {
                'title': 'Cluster A',
                'sentence_count': 5,
                'sentiment': 'positive'
            },
            {
                'title': 'Cluster B',
                'sentence_count': 5,
                'sentiment': 'negative'
            }
        ]
        
        summary = generator.generate_overall_summary(clusters)
        # With tie, first max wins (depends on Python version, but should be consistent)
        assert summary['dominant_sentiment'] in ['positive', 'negative']
        assert summary['total_sentences'] == 10
        assert summary['sentiment_distribution']['positive'] == 5
        assert summary['sentiment_distribution']['negative'] == 5
    
    def test_generate_overall_summary_mixed_sentiment(self):
        """Test overall summary with mixed sentiment in same cluster."""
        generator = InsightGenerator()
        clusters = [
            {
                'title': 'Mixed Cluster',
                'sentence_count': 8,
                'sentiment': 'mixed'
            },
            {
                'title': 'Positive Cluster',
                'sentence_count': 4,
                'sentiment': 'positive'
            }
        ]
        
        summary = generator.generate_overall_summary(clusters)
        assert summary['total_sentences'] == 12
        assert summary['sentiment_distribution']['mixed'] == 8
        assert summary['sentiment_distribution']['positive'] == 4
        assert summary['dominant_sentiment'] == 'mixed'
    
    def test_summary_deterministic(self):
        """Test that summary generation is deterministic."""
        generator = InsightGenerator()
        sentences = ["Test sentence"] * 7
        sentiment_info = {'sentiment': 'positive'}
        
        summary1 = generator._generate_summary(sentences, sentiment_info)
        summary2 = generator._generate_summary(sentences, sentiment_info)
        
        assert summary1 == summary2
    
    def test_overall_summary_deterministic(self):
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
        
        assert summary1 == summary2
    
    @pytest.mark.skip(reason="Production-specific summary logic not implemented yet")
    def test_custom_summary_templates(self):
        """TODO: Test custom summary templates for production."""
        pass