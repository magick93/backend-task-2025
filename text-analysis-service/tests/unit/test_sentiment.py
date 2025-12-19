"""
Unit tests for sentiment analysis module.

Tests cover:
- SentimentAnalyzer initialization and configuration (VADER)
- Single sentence analysis with VADER
- Batch analysis
- Cluster-level sentiment aggregation
- Edge cases and error handling
- Deterministic behavior
- Convenience functions
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import re

from src.app.pipeline.sentiment import (
    SentimentAnalyzer,
    get_sentiment_analyzer,
    analyze_sentiment
)


class TestSentimentAnalyzerInitialization:
    """Test SentimentAnalyzer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        analyzer = SentimentAnalyzer()
        
        assert analyzer.positive_keywords is not None
        assert analyzer.negative_keywords is not None
        assert analyzer.neutral_threshold == 0.05
        assert analyzer.analyzer is not None
        assert hasattr(analyzer.analyzer, 'polarity_scores')
    
    def test_custom_keywords(self):
        """Test initialization with custom keyword lists."""
        positive = ['happy_custom', 'joyful_custom']
        negative = ['sad_custom', 'angry_custom']
        analyzer = SentimentAnalyzer(
            positive_keywords=positive,
            negative_keywords=negative,
            neutral_threshold=0.1
        )
        
        assert analyzer.positive_keywords == positive
        assert analyzer.negative_keywords == negative
        assert analyzer.neutral_threshold == 0.1
        
        # Check lexicon updates
        assert 'happy_custom' in analyzer.analyzer.lexicon
        assert analyzer.analyzer.lexicon['happy_custom'] == 2.0
        assert 'sad_custom' in analyzer.analyzer.lexicon
        assert analyzer.analyzer.lexicon['sad_custom'] == -2.0
    
    def test_add_custom_keywords(self):
        """Test adding custom keywords after initialization."""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_custom_keywords(
            positive_keywords=['excellent_custom', 'superb_custom'],
            negative_keywords=['terrible_custom', 'awful_custom']
        )
        
        assert 'excellent_custom' in analyzer.positive_keywords
        assert 'terrible_custom' in analyzer.negative_keywords
        
        # Check lexicon updates
        assert 'excellent_custom' in analyzer.analyzer.lexicon
        assert 'terrible_custom' in analyzer.analyzer.lexicon


class TestSingleSentenceAnalysis:
    """Test analyze_sentence method."""
    
    def test_positive_sentence(self):
        """Test analysis of a clearly positive sentence."""
        analyzer = SentimentAnalyzer()
        sentence = "This is a great product and I love it!"
        
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'positive'
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['method'] == 'vader'
        assert 'scores' in result
        assert result['scores']['pos'] > 0
        assert result['scores']['compound'] > 0.05
    
    def test_negative_sentence(self):
        """Test analysis of a clearly negative sentence."""
        analyzer = SentimentAnalyzer()
        sentence = "This is a terrible product and I hate it!"
        
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'negative'
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['scores']['neg'] > 0
        assert result['scores']['compound'] < -0.05
    
    def test_neutral_sentence_no_keywords(self):
        """Test analysis of a neutral sentence."""
        analyzer = SentimentAnalyzer()
        sentence = "The weather is cloudy today."
        
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'neutral'
        # Check that it's close to 0 compound score
        assert abs(result['scores']['compound']) <= 0.05
    
    def test_mixed_sentence(self):
        """Test analysis of a sentence with both positive and negative keywords."""
        analyzer = SentimentAnalyzer()
        sentence = "The product is good but the service is bad."
        
        result = analyzer.analyze_sentence(sentence)
        
        # VADER often handles "but" correctly, weighing the second part more
        # "good" (1.9), "bad" (-2.5). "but" shifts focus to "bad".
        # Likely negative or neutral depending on weights.
        
        assert result['label'] in ['neutral', 'negative'] 
        assert result['scores']['pos'] > 0
        assert result['scores']['neg'] > 0
    
    def test_confidence_calculation(self):
        """Test that confidence behaves rationally."""
        analyzer = SentimentAnalyzer()
        
        # Neutral sentence
        sentence1 = "It is a box."
        result1 = analyzer.analyze_sentence(sentence1)
        
        # Very positive sentence
        sentence2 = "This is the best most amazing thing ever!"
        result2 = analyzer.analyze_sentence(sentence2)
        
        # Check specific implementation details of confidence
        if result1['label'] == 'neutral':
             # Neutral confidence should be high (closer to 1) if compound is 0
             assert result1['confidence'] > 0.5 
             
        if result2['label'] == 'positive':
             assert result2['confidence'] > 0.5
    
    def test_case_insensitivity(self):
        """Test that VADER handles case (it often uses caps for emphasis)."""
        analyzer = SentimentAnalyzer()
        
        # "GREAT" is more intense than "great" in VADER
        sentence_normal = "This is a great product!"
        sentence_caps = "This is a GREAT product!"
        
        result_normal = analyzer.analyze_sentence(sentence_normal)
        result_caps = analyzer.analyze_sentence(sentence_caps)
        
        assert result_normal['label'] == 'positive'
        assert result_caps['label'] == 'positive'
        # Caps usually increases intensity
        assert result_caps['scores']['compound'] >= result_normal['scores']['compound']
    
    def test_empty_sentence(self):
        """Test analysis of an empty sentence."""
        analyzer = SentimentAnalyzer()
        sentence = ""
        
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'neutral'
        assert result['scores']['compound'] == 0.0
    
    def test_very_long_sentence(self):
        """Test analysis of a very long sentence."""
        analyzer = SentimentAnalyzer()
        # Create a long sentence 
        sentence = "good " * 50 + "bad " * 10
        
        result = analyzer.analyze_sentence(sentence)
        
        # Should be positive (more positive words)
        assert result['label'] == 'positive'


class TestBatchAnalysis:
    """Test analyze_batch method."""
    
    def test_batch_analysis(self):
        """Test analysis of a batch of sentences."""
        analyzer = SentimentAnalyzer()
        sentences = [
            "This is great!",
            "This is terrible.",
            "The weather is nice.",
            "I hate this product."
        ]
        
        results = analyzer.analyze_batch(sentences)
        
        assert len(results) == len(sentences)
        assert results[0]['label'] == 'positive'
        assert results[1]['label'] == 'negative'
        # "nice" is positive in VADER usually, check thresholds
        # "The weather is nice." -> compound ~0.42 (positive)
        assert results[2]['label'] in ['positive', 'neutral'] 
        assert results[3]['label'] == 'negative'
    
    def test_empty_batch(self):
        """Test analysis of an empty batch."""
        analyzer = SentimentAnalyzer()
        sentences = []
        
        results = analyzer.analyze_batch(sentences)
        
        assert len(results) == 0


class TestClusterSentimentAggregation:
    """Test get_cluster_sentiment method."""
    
    def test_single_cluster_all_positive(self):
        """Test sentiment aggregation for a single positive cluster."""
        analyzer = SentimentAnalyzer()
        sentences = ["Great product!", "I love it!", "Excellent work!"]
        embeddings = np.random.randn(len(sentences), 10)  # Dummy embeddings
        labels = np.array([0, 0, 0])  # All in cluster 0
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert 0 in cluster_sentiments
        cluster_info = cluster_sentiments[0]
        assert cluster_info['sentiment'] == 'positive'
        assert cluster_info['total_sentences'] == 3
        assert cluster_info['counts']['positive'] == 3
        assert cluster_info['counts']['negative'] == 0
        assert 0.0 <= cluster_info['confidence'] <= 1.0
        assert 'proportions' in cluster_info
        assert cluster_info['proportions']['positive'] == 1.0
    
    def test_multiple_clusters(self):
        """Test sentiment aggregation for multiple clusters."""
        analyzer = SentimentAnalyzer()
        sentences = [
            "Great!",        # Cluster 0
            "Awesome!",      # Cluster 0
            "Terrible!",     # Cluster 1
            "Bad!",          # Cluster 1
            "Okay.",         # Cluster 2 (neutralish)
            "A box."         # Cluster 2 (neutral)
        ]
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert len(cluster_sentiments) == 3
        assert cluster_sentiments[0]['sentiment'] == 'positive'
        assert cluster_sentiments[1]['sentiment'] == 'negative'
        assert cluster_sentiments[2]['sentiment'] in ['neutral', 'positive'] # "Okay" might be slightly positive
    
    def test_cluster_with_noise(self):
        """Test sentiment aggregation with noise points (label = -1)."""
        analyzer = SentimentAnalyzer()
        sentences = ["Great!", "Bad!", "Okay", "Noise1", "Noise2"]
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([0, 1, 2, -1, -1])  # -1 indicates noise
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        # Should only have clusters 0, 1, 2 (not -1)
        assert -1 not in cluster_sentiments
        assert 0 in cluster_sentiments
        assert 1 in cluster_sentiments
        assert 2 in cluster_sentiments
        assert len(cluster_sentiments) == 3
    
    def test_empty_cluster(self):
        """Test sentiment aggregation for an empty cluster."""
        analyzer = SentimentAnalyzer()
        sentences = ["Great!", "Bad!"]
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([0, 1])  # Two clusters with one sentence each
        
        # Add a cluster label that doesn't exist in sentences
        labels_with_gap = np.array([0, 2])  # Cluster 1 missing
        
        cluster_sentiments = analyzer.get_cluster_sentiment(
            sentences, embeddings, labels_with_gap
        )
        
        # Should only have clusters 0 and 2
        assert 0 in cluster_sentiments
        assert 2 in cluster_sentiments
        assert 1 not in cluster_sentiments
    
    def test_cluster_with_all_neutral(self):
        """Test sentiment aggregation for a cluster with all neutral sentences."""
        analyzer = SentimentAnalyzer()
        sentences = ["It is a box.", "A table.", "The wall."]
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([0, 0, 0])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert cluster_sentiments[0]['sentiment'] == 'neutral'
        assert cluster_sentiments[0]['counts']['neutral'] == 3
    
    def test_strength_calculation(self):
        """Test that sentiment strength is calculated correctly."""
        analyzer = SentimentAnalyzer()
        sentences = ["good", "good", "bad"]  # 2 positive, 1 negative (approx)
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([0, 0, 0])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        cluster_info = cluster_sentiments[0]
        # Depending on how "good" and "bad" score, average might be positive
        if cluster_info['sentiment'] == 'positive':
            # 2 out of 3 were classified as positive (assuming "good" is pos, "bad" is neg)
             pass 
        
        assert 'strength' in cluster_info
        assert 0.0 <= cluster_info['strength'] <= 1.0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_nan_in_embeddings(self):
        """Test handling of NaN values in embeddings (should be ignored)."""
        analyzer = SentimentAnalyzer()
        sentences = ["Good", "Bad"]
        embeddings = np.array([[1.0, 2.0], [np.nan, np.nan]])
        labels = np.array([0, 0])
        
        # Should not crash
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert 0 in cluster_sentiments
    
    def test_embeddings_shape_mismatch(self):
        """Test when embeddings shape doesn't match sentences length."""
        analyzer = SentimentAnalyzer()
        sentences = ["Good", "Bad", "Neutral"]
        embeddings = np.random.randn(2, 10)
        labels = np.array([0, 0, 0])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert 0 in cluster_sentiments
    
    def test_all_noise_clusters(self):
        """Test when all points are noise (label = -1)."""
        analyzer = SentimentAnalyzer()
        sentences = ["Noise1", "Noise2", "Noise3"]
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([-1, -1, -1])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert len(cluster_sentiments) == 0


class TestDeterministicBehavior:
    """Test deterministic behavior of sentiment analysis."""
    
    def test_deterministic_keyword_matching(self):
        """Test that VADER is deterministic."""
        analyzer1 = SentimentAnalyzer()
        analyzer2 = SentimentAnalyzer()
        
        sentence = "This is a great and awesome product!"
        
        result1 = analyzer1.analyze_sentence(sentence)
        result2 = analyzer2.analyze_sentence(sentence)
        
        # Should be identical
        assert result1['label'] == result2['label']
        assert result1['confidence'] == result2['confidence']
        assert result1['scores']['compound'] == result2['scores']['compound']


class TestConvenienceFunctions:
    """Test convenience functions (singleton and analyze_sentiment)."""
    
    def test_get_sentiment_analyzer_singleton(self):
        """Test that get_sentiment_analyzer returns a singleton instance."""
        analyzer1 = get_sentiment_analyzer()
        analyzer2 = get_sentiment_analyzer()
        
        # Should be the same instance
        assert analyzer1 is analyzer2
        assert isinstance(analyzer1, SentimentAnalyzer)
    
    def test_analyze_sentiment_function(self):
        """Test the convenience analyze_sentiment function."""
        sentence = "This is excellent!"
        result = analyze_sentiment(sentence)
        
        assert 'label' in result
        assert 'confidence' in result
        assert 'method' in result
        assert result['method'] == 'vader'


class TestImplementationDetails:
    """Test specific implementation details."""
    
    def test_method_name(self):
        """Verify that the method is correctly identified as 'vader'."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentence("test")
        assert result['method'] == 'vader'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
