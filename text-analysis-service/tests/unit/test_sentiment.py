"""
Unit tests for sentiment analysis module.

Tests cover:
- SentimentAnalyzer initialization and configuration
- Single sentence analysis with keyword matching
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
        assert len(analyzer.positive_keywords) > 0
        assert len(analyzer.negative_keywords) > 0
        assert analyzer.neutral_threshold == 0.1
        assert analyzer.positive_pattern is not None
        assert analyzer.negative_pattern is not None
    
    def test_custom_keywords(self):
        """Test initialization with custom keyword lists."""
        positive = ['happy', 'joyful']
        negative = ['sad', 'angry']
        analyzer = SentimentAnalyzer(
            positive_keywords=positive,
            negative_keywords=negative,
            neutral_threshold=0.2
        )
        
        assert analyzer.positive_keywords == positive
        assert analyzer.negative_keywords == negative
        assert analyzer.neutral_threshold == 0.2
        
        # Patterns should be compiled
        assert 'happy' in analyzer.positive_pattern.pattern
        assert 'sad' in analyzer.negative_pattern.pattern
    
    def test_add_custom_keywords(self):
        """Test adding custom keywords after initialization."""
        analyzer = SentimentAnalyzer()
        original_positive_count = len(analyzer.positive_keywords)
        original_negative_count = len(analyzer.negative_keywords)
        
        analyzer.add_custom_keywords(
            positive_keywords=['excellent', 'superb'],
            negative_keywords=['terrible', 'awful']
        )
        
        assert len(analyzer.positive_keywords) == original_positive_count + 2
        assert len(analyzer.negative_keywords) == original_negative_count + 2
        assert 'excellent' in analyzer.positive_keywords
        assert 'terrible' in analyzer.negative_keywords
        
        # Patterns should be recompiled
        assert 'excellent' in analyzer.positive_pattern.pattern
        assert 'terrible' in analyzer.negative_pattern.pattern


class TestSingleSentenceAnalysis:
    """Test analyze_sentence method."""
    
    def test_positive_sentence(self):
        """Test analysis of a clearly positive sentence."""
        analyzer = SentimentAnalyzer()
        sentence = "This is a great product and I love it!"
        
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'positive'
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['method'] == 'keyword_heuristic'
        assert 'scores' in result
        assert result['scores']['positive_matches'] >= 1
        assert result['scores']['negative_matches'] == 0
    
    def test_negative_sentence(self):
        """Test analysis of a clearly negative sentence."""
        analyzer = SentimentAnalyzer()
        sentence = "This is a terrible product and I hate it!"
        
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'negative'
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['scores']['negative_matches'] >= 1
        assert result['scores']['positive_matches'] == 0
    
    def test_neutral_sentence_no_keywords(self):
        """Test analysis of a neutral sentence with no keywords."""
        analyzer = SentimentAnalyzer()
        sentence = "The weather is cloudy today."
        
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'neutral'
        assert result['confidence'] == 0.5  # Default confidence for neutral
        assert result['scores']['positive_matches'] == 0
        assert result['scores']['negative_matches'] == 0
    
    def test_mixed_sentence(self):
        """Test analysis of a sentence with both positive and negative keywords."""
        analyzer = SentimentAnalyzer()
        sentence = "The product is good but the service is bad."
        
        result = analyzer.analyze_sentence(sentence)
        
        # Should be neutral if positive and negative matches are balanced
        # With default neutral_threshold=0.1, difference < 0.1 -> neutral
        # positive_matches=1, negative_matches=1, total=2, diff=0 -> neutral
        assert result['label'] == 'neutral'
        assert result['scores']['positive_matches'] == 1
        assert result['scores']['negative_matches'] == 1
    
    def test_confidence_calculation(self):
        """Test that confidence increases with more keyword matches."""
        analyzer = SentimentAnalyzer()
        
        # Single positive keyword
        sentence1 = "Good"
        result1 = analyzer.analyze_sentence(sentence1)
        
        # Multiple positive keywords
        sentence2 = "Good great excellent fantastic"
        result2 = analyzer.analyze_sentence(sentence2)
        
        # Confidence should be higher or equal with more matches
        # (capped at 0.95, so may be equal if first already at cap)
        assert result2['confidence'] >= result1['confidence']
        assert result2['confidence'] <= 0.95  # Capped at 0.95
    
    def test_case_insensitivity(self):
        """Test that keyword matching is case-insensitive."""
        analyzer = SentimentAnalyzer()
        
        # Mixed case
        sentence = "This is a GREAT product!"
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'positive'
        assert result['scores']['positive_matches'] >= 1
    
    def test_word_boundaries(self):
        """Test that keyword matching respects word boundaries."""
        analyzer = SentimentAnalyzer()
        
        # "good" should not match "goodness"
        sentence = "The goodness of the product"
        result = analyzer.analyze_sentence(sentence)
        
        # "good" is not a separate word, so should not match
        assert result['scores']['positive_matches'] == 0
        assert result['label'] == 'neutral'
    
    def test_empty_sentence(self):
        """Test analysis of an empty sentence."""
        analyzer = SentimentAnalyzer()
        sentence = ""
        
        result = analyzer.analyze_sentence(sentence)
        
        assert result['label'] == 'neutral'
        assert result['confidence'] == 0.5
        assert result['scores']['positive_matches'] == 0
        assert result['scores']['negative_matches'] == 0
    
    def test_very_long_sentence(self):
        """Test analysis of a very long sentence."""
        analyzer = SentimentAnalyzer()
        # Create a long sentence with many positive keywords
        sentence = "good " * 50 + "bad " * 10
        
        result = analyzer.analyze_sentence(sentence)
        
        # Should be positive (more positive matches)
        assert result['label'] == 'positive'
        assert result['scores']['positive_matches'] == 50
        assert result['scores']['negative_matches'] == 10


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
        assert results[2]['label'] == 'neutral'  # No keywords
        assert results[3]['label'] == 'negative'
    
    def test_empty_batch(self):
        """Test analysis of an empty batch."""
        analyzer = SentimentAnalyzer()
        sentences = []
        
        results = analyzer.analyze_batch(sentences)
        
        assert len(results) == 0
    
    def test_batch_with_mixed_sentiments(self):
        """Test batch with mixed sentiments."""
        analyzer = SentimentAnalyzer()
        sentences = ["good"] * 3 + ["bad"] * 2 + ["neutral sentence"] * 1
        
        results = analyzer.analyze_batch(sentences)
        
        positive_count = sum(1 for r in results if r['label'] == 'positive')
        negative_count = sum(1 for r in results if r['label'] == 'negative')
        neutral_count = sum(1 for r in results if r['label'] == 'neutral')
        
        assert positive_count == 3
        assert negative_count == 2
        assert neutral_count == 1


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
        assert cluster_info['counts']['neutral'] == 0
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
            "Okay.",         # Cluster 2 (neutral)
            "The sky is blue."  # Cluster 2 (neutral)
        ]
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert len(cluster_sentiments) == 3
        assert cluster_sentiments[0]['sentiment'] == 'positive'
        assert cluster_sentiments[1]['sentiment'] == 'negative'
        assert cluster_sentiments[2]['sentiment'] == 'neutral'
    
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
        # This shouldn't happen in practice, but test robustness
        labels_with_gap = np.array([0, 2])  # Cluster 1 missing
        
        cluster_sentiments = analyzer.get_cluster_sentiment(
            sentences, embeddings, labels_with_gap
        )
        
        # Should only have clusters 0 and 2
        assert 0 in cluster_sentiments
        assert 2 in cluster_sentiments
        assert 1 not in cluster_sentiments
    
    def test_tie_breaking_with_confidence(self):
        """Test tie-breaking when sentiment counts are equal."""
        analyzer = SentimentAnalyzer()
        # Create sentences where positive and negative counts are equal
        # but confidences differ
        sentences = ["good", "bad"]  # Both have 1 keyword, equal counts
        
        # Mock analyze_sentence to control confidence values
        with patch.object(analyzer, 'analyze_sentence') as mock_analyze:
            mock_analyze.side_effect = [
                {'label': 'positive', 'confidence': 0.9},  # High confidence positive
                {'label': 'negative', 'confidence': 0.6}   # Lower confidence negative
            ]
            
            embeddings = np.random.randn(len(sentences), 10)
            labels = np.array([0, 0])  # Both in same cluster
            
            cluster_sentiments = analyzer.get_cluster_sentiment(
                sentences, embeddings, labels
            )
            
            # Should choose positive because higher average confidence
            assert cluster_sentiments[0]['sentiment'] == 'positive'
    
    def test_cluster_with_all_neutral(self):
        """Test sentiment aggregation for a cluster with all neutral sentences."""
        analyzer = SentimentAnalyzer()
        sentences = ["The weather is nice.", "It is raining.", "Cloudy today."]
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([0, 0, 0])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert cluster_sentiments[0]['sentiment'] == 'neutral'
        assert cluster_sentiments[0]['counts']['neutral'] == 3
        assert cluster_sentiments[0]['counts']['positive'] == 0
        assert cluster_sentiments[0]['counts']['negative'] == 0
    
    def test_strength_calculation(self):
        """Test that sentiment strength is calculated correctly."""
        analyzer = SentimentAnalyzer()
        sentences = ["good", "good", "bad"]  # 2 positive, 1 negative
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([0, 0, 0])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        cluster_info = cluster_sentiments[0]
        assert cluster_info['sentiment'] == 'positive'
        # Strength should be proportion of dominant sentiment
        assert cluster_info['strength'] == pytest.approx(2/3, rel=1e-3)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_nan_in_embeddings(self):
        """Test handling of NaN values in embeddings (should be ignored)."""
        analyzer = SentimentAnalyzer()
        sentences = ["Good", "Bad"]
        embeddings = np.array([[1.0, 2.0], [np.nan, np.nan]])  # Second embedding has NaN
        labels = np.array([0, 0])
        
        # Should not crash
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        # Should still process the sentences
        assert 0 in cluster_sentiments
    
    def test_embeddings_shape_mismatch(self):
        """Test when embeddings shape doesn't match sentences length."""
        analyzer = SentimentAnalyzer()
        sentences = ["Good", "Bad", "Neutral"]
        embeddings = np.random.randn(2, 10)  # Only 2 embeddings for 3 sentences
        labels = np.array([0, 0, 0])
        
        # This should still work (embeddings are unused in current implementation)
        # but we test it doesn't crash
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert 0 in cluster_sentiments
    
    def test_all_noise_clusters(self):
        """Test when all points are noise (label = -1)."""
        analyzer = SentimentAnalyzer()
        sentences = ["Noise1", "Noise2", "Noise3"]
        embeddings = np.random.randn(len(sentences), 10)
        labels = np.array([-1, -1, -1])
        
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        # Should return empty dict (no valid clusters)
        assert len(cluster_sentiments) == 0


class TestDeterministicBehavior:
    """Test deterministic behavior of sentiment analysis."""
    
    def test_deterministic_keyword_matching(self):
        """Test that keyword matching is deterministic."""
        analyzer1 = SentimentAnalyzer()
        analyzer2 = SentimentAnalyzer()
        
        sentence = "This is a great and awesome product!"
        
        result1 = analyzer1.analyze_sentence(sentence)
        result2 = analyzer2.analyze_sentence(sentence)
        
        # Should be identical
        assert result1['label'] == result2['label']
        assert result1['confidence'] == result2['confidence']
        assert result1['scores']['positive_matches'] == result2['scores']['positive_matches']
        assert result1['scores']['negative_matches'] == result2['scores']['negative_matches']
    
    def test_deterministic_across_multiple_calls(self):
        """Test that same analyzer produces same results across multiple calls."""
        analyzer = SentimentAnalyzer()
        sentence = "This is good and bad at the same time."
        
        result1 = analyzer.analyze_sentence(sentence)
        result2 = analyzer.analyze_sentence(sentence)
        
        assert result1['label'] == result2['label']
        assert result1['confidence'] == result2['confidence']


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
        assert result['method'] == 'keyword_heuristic'
    
    def test_analyze_sentiment_with_different_sentences(self):
        """Test analyze_sentiment with various sentence types."""
        # Positive
        result1 = analyze_sentiment("Great!")
        assert result1['label'] == 'positive'
        
        # Negative
        result2 = analyze_sentiment("Terrible!")
        assert result2['label'] == 'negative'
        
        # Neutral
        result3 = analyze_sentiment("The weather is cloudy.")
        assert result3['label'] == 'neutral'


class TestTODOComments:
    """Test areas marked for future improvement (TODO comments)."""
    
    def test_todo_placeholder_implementation(self):
        """Verify that the current implementation is a placeholder."""
        analyzer = SentimentAnalyzer()
        
        # Check that method is keyword_heuristic (not a production model)
        result = analyzer.analyze_sentence("test")
        assert result['method'] == 'keyword_heuristic'
        
        # TODO: In production, this should be replaced with proper sentiment analysis
        # (VADER, transformers, etc.)
    
    def test_embeddings_unused(self):
        """Test that embeddings parameter is currently unused (as per TODO)."""
        analyzer = SentimentAnalyzer()
        sentences = ["Good", "Bad"]
        embeddings = np.random.randn(2, 384)
        labels = np.array([0, 0])
        
        # This should work even though embeddings are unused
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        # TODO: In production, embeddings could be used for more sophisticated analysis
        assert 0 in cluster_sentiments


class TestProductionReadiness:
    """Test considerations for production readiness."""
    
    def test_performance_with_large_batch(self):
        """Test that batch analysis doesn't crash with large input."""
        analyzer = SentimentAnalyzer()
        sentences = ["This is sentence " + str(i) for i in range(1000)]
        
        # Should complete without error
        results = analyzer.analyze_batch(sentences)
        assert len(results) == 1000
        
        # All results should have expected structure
        for result in results:
            assert 'label' in result
            assert result['label'] in ['positive', 'negative', 'neutral']
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_memory_usage_with_large_clusters(self):
        """Test memory usage with large clusters."""
        analyzer = SentimentAnalyzer()
        
        # Create many sentences with embeddings
        n_sentences = 500
        sentences = ["Good" if i % 2 == 0 else "Bad" for i in range(n_sentences)]
        embeddings = np.random.randn(n_sentences, 10)
        labels = np.array([0] * n_sentences)  # All in same cluster
        
        # Should complete without memory issues
        cluster_sentiments = analyzer.get_cluster_sentiment(sentences, embeddings, labels)
        
        assert 0 in cluster_sentiments
        assert cluster_sentiments[0]['total_sentences'] == n_sentences
    
    def test_thread_safety(self):
        """Test that analyzer can be used from multiple threads (basic check)."""
        import threading
        
        analyzer = SentimentAnalyzer()
        results = []
        
        def analyze_thread(sentence):
            result = analyzer.analyze_sentence(sentence)
            results.append(result)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=analyze_thread, args=(f"Sentence {i} good",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 10
        # All should have valid structure
        for result in results:
            assert 'label' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])