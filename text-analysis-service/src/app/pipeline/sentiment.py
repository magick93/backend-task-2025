"""
Simple sentiment analysis module.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis.
Aggregate sentiment at cluster level.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using VADER.
    """
    
    def __init__(
        self,
        positive_keywords: Optional[List[str]] = None,
        negative_keywords: Optional[List[str]] = None,
        neutral_threshold: float = 0.05
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            positive_keywords: List of additional positive sentiment keywords
            negative_keywords: List of additional negative sentiment keywords  
            neutral_threshold: Threshold for neutral sentiment (compound score absolute value)
                               Default is 0.05 as per VADER recommendations.
        """
        self.analyzer = VaderAnalyzer()
        self.neutral_threshold = neutral_threshold
        
        # Keep track of custom keywords for reference
        self.positive_keywords = []
        self.negative_keywords = []
        
        # Add custom keywords if provided
        if positive_keywords or negative_keywords:
            self.add_custom_keywords(positive_keywords or [], negative_keywords or [])
            
        logger.debug("SentimentAnalyzer initialized with VADER")
    
    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single sentence using VADER.
        
        Args:
            sentence: Text sentence to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not sentence:
            return {
                'label': 'neutral',
                'confidence': 0.0,
                'scores': {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0},
                'method': 'vader'
            }

        scores = self.analyzer.polarity_scores(sentence)
        compound = scores['compound']
        
        if compound >= self.neutral_threshold:
            sentiment = 'positive'
        elif compound <= -self.neutral_threshold:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        # Use the probability of the dominant class as confidence
        # For neutral, use 'neu'. For pos/neg, use 'pos'/'neg'?
        # VADER scores (neg, neu, pos) sum to 1.
        # However, compound is often the best single metric.
        # Let's use max(neg, neu, pos) as a proxy for confidence in that specific classification component
        # OR just use the absolute value of compound for intensity confidence?
        # The prompt asked for specific mapping logic.
        
        # Let's try to align confidence with the classification.
        if sentiment == 'positive':
            confidence = scores['pos']
            # If compound is high but pos is low (e.g. mostly neutral but slightly positive), 
            # maybe compound is better?
            # VADER docs say compound is the "normalized, weighted composite score".
            # Let's use simple logic:
            # If we say it's positive, how positive? -> compound.
            # But compound can be 0.9 even if 'pos' is 0.4 (if text is long?).
            
            # Let's stick to what usually works:
            # For this task, let's use max(neg, neu, pos) as confidence that it belongs to that bucket
            # relative to the text content.
            confidence = scores['pos']
            if confidence == 0: # Fallback if compound triggered but pos component is 0 (rare but possible with custom lexicons)
                 confidence = abs(compound)
                 
        elif sentiment == 'negative':
            confidence = scores['neg']
            if confidence == 0:
                confidence = abs(compound)
        else:
            confidence = scores['neu']
            
        # Actually, standard VADER usage often just uses compound. 
        # But 'confidence' implies probability.
        # Let's use a simpler heuristic for confidence:
        # If Positive/Negative: abs(compound)
        # If Neutral: 1 - abs(compound)  (i.e. closeness to 0)
        
        if sentiment == 'neutral':
             confidence = 1.0 - abs(compound)
        else:
             confidence = abs(compound)

        return {
            'label': sentiment,
            'confidence': float(confidence),
            'scores': scores,
            'method': 'vader'
        }
    
    def analyze_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of sentences.
        
        Args:
            sentences: List of text sentences
            
        Returns:
            List of sentiment analysis results
        """
        logger.debug(f"Analyzing sentiment for {len(sentences)} sentences")
        return [self.analyze_sentence(sentence) for sentence in sentences]
    
    def get_cluster_sentiment(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        precomputed_sentiments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculate aggregate sentiment for each cluster.
        
        Args:
            sentences: List of all sentences
            embeddings: Sentence embeddings (unused in this implementation)
            labels: Cluster labels (-1 for noise)
            precomputed_sentiments: Optional list of pre-computed sentiment results matching sentences order
            
        Returns:
            Dictionary mapping cluster label to sentiment analysis
        """
        cluster_sentiments = {}
        unique_labels = np.unique(labels)
        
        logger.debug(f"Aggregating sentiment for {len(unique_labels)} clusters")
        
        # Analyze all sentences once if not precomputed
        if precomputed_sentiments is None:
            sentiment_results_map = self.analyze_batch(sentences)
        else:
            sentiment_results_map = precomputed_sentiments
            
        if len(sentiment_results_map) != len(sentences):
             logger.warning(f"Mismatch between sentences ({len(sentences)}) and sentiment results ({len(sentiment_results_map)})")
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            
            cluster_indices = np.where(labels == label)[0]
            
            # Gather results for this cluster
            cluster_results = []
            for i in cluster_indices:
                if i < len(sentiment_results_map):
                    cluster_results.append(sentiment_results_map[i])
            
            if not cluster_results:
                continue
            
            # Aggregate sentiment
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            compound_scores = []
            
            for result in cluster_results:
                sentiment_counts[result['label']] += 1
                compound_scores.append(result['scores']['compound'])
            
            # Calculate average compound score for the cluster
            avg_compound = np.mean(compound_scores) if compound_scores else 0.0
            
            # Determine overall sentiment based on average compound
            if avg_compound >= self.neutral_threshold:
                overall_sentiment = 'positive'
            elif avg_compound <= -self.neutral_threshold:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            # Calculate average confidence (using the logic from analyze_sentence)
            if overall_sentiment == 'neutral':
                avg_confidence = 1.0 - abs(avg_compound)
            else:
                avg_confidence = abs(avg_compound)
                
            total = len(cluster_results)
            proportions = {k: v/total for k, v in sentiment_counts.items()}
            
            # Calculate sentiment strength (proportion of the dominant sentiment)
            sentiment_strength = proportions.get(overall_sentiment, 0)
            
            cluster_sentiments[label] = {
                'sentiment': overall_sentiment,
                'confidence': float(avg_confidence),
                'strength': float(sentiment_strength),
                'proportions': proportions,
                'counts': sentiment_counts,
                'total_sentences': total,
                'avg_compound': float(avg_compound)
            }
        
        return cluster_sentiments
    
    def add_custom_keywords(
        self, 
        positive_keywords: List[str], 
        negative_keywords: List[str]
    ) -> None:
        """
        Add custom keywords to the VADER lexicon.
        
        Args:
            positive_keywords: Additional positive keywords
            negative_keywords: Additional negative keywords
        """
        new_words = {}
        for word in positive_keywords:
            new_words[word] = 2.0  # Arbitrary positive score
        for word in negative_keywords:
            new_words[word] = -2.0  # Arbitrary negative score
            
        self.analyzer.lexicon.update(new_words)
        
        # Keep track
        self.positive_keywords.extend(positive_keywords)
        self.negative_keywords.extend(negative_keywords)
        
        logger.debug(
            f"Added {len(positive_keywords)} positive and "
            f"{len(negative_keywords)} negative keywords to VADER lexicon"
        )


# Singleton instance for easy import
_default_sentiment_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """
    Get or create the default sentiment analyzer instance.
    
    Returns:
        Shared SentimentAnalyzer instance
    """
    global _default_sentiment_analyzer
    if _default_sentiment_analyzer is None:
        _default_sentiment_analyzer = SentimentAnalyzer()
        logger.info("Created default sentiment analyzer instance")
    
    return _default_sentiment_analyzer


def analyze_sentiment(sentence: str) -> Dict[str, Any]:
    """
    Convenience function to analyze sentiment of a single sentence.
    
    Args:
        sentence: Text sentence to analyze
        
    Returns:
        Sentiment analysis results
    """
    analyzer = get_sentiment_analyzer()
    return analyzer.analyze_sentence(sentence)
