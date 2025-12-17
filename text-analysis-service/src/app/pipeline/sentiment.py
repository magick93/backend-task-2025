"""
Simple sentiment analysis module.

Either lightweight heuristic or stub with clear TODO.
Aggregate sentiment at cluster level.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class SentimentAnalyzer:
    """
    Simple sentiment analyzer using heuristic rules.
    
    This is a placeholder implementation - focus is on clarity and
    deterministic behavior rather than production-quality sentiment analysis.
    TODO: Replace with proper sentiment analysis (VADER, transformers, etc.)
    """
    
    def __init__(
        self,
        positive_keywords: Optional[List[str]] = None,
        negative_keywords: Optional[List[str]] = None,
        neutral_threshold: float = 0.1
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            positive_keywords: List of positive sentiment keywords
            negative_keywords: List of negative sentiment keywords  
            neutral_threshold: Threshold for neutral sentiment confidence
        """
        self.positive_keywords = positive_keywords or [
            'good', 'great', 'excellent', 'awesome', 'fantastic',
            'love', 'like', 'happy', 'pleased', 'satisfied',
            'positive', 'wonderful', 'amazing', 'best', 'perfect'
        ]
        
        self.negative_keywords = negative_keywords or [
            'bad', 'poor', 'terrible', 'awful', 'horrible',
            'hate', 'dislike', 'unhappy', 'angry', 'frustrated',
            'negative', 'worst', 'disappointed', 'failed', 'broken'
        ]
        
        self.neutral_threshold = neutral_threshold
        
        # Compile regex patterns for faster matching
        self.positive_pattern = re.compile(
            r'\b(' + '|'.join(self.positive_keywords) + r')\b',
            re.IGNORECASE
        )
        self.negative_pattern = re.compile(
            r'\b(' + '|'.join(self.negative_keywords) + r')\b',
            re.IGNORECASE
        )
        
        logger.debug(
            f"SentimentAnalyzer initialized with {len(self.positive_keywords)} "
            f"positive and {len(self.negative_keywords)} negative keywords"
        )
    
    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single sentence using keyword matching.
        
        Args:
            sentence: Text sentence to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # TODO: Replace with proper sentiment analysis
        # This is a simple heuristic implementation
        
        sentence_lower = sentence.lower()
        
        # Count keyword matches
        positive_matches = len(self.positive_pattern.findall(sentence_lower))
        negative_matches = len(self.negative_pattern.findall(sentence_lower))
        
        # Calculate sentiment scores
        total_matches = positive_matches + negative_matches
        
        if total_matches == 0:
            # No keywords found - default to neutral
            sentiment = 'neutral'
            confidence = 0.5
        else:
            positive_score = positive_matches / total_matches
            negative_score = negative_matches / total_matches
            
            if abs(positive_score - negative_score) < self.neutral_threshold:
                sentiment = 'neutral'
                confidence = 0.5
            elif positive_score > negative_score:
                sentiment = 'positive'
                confidence = positive_score
            else:
                sentiment = 'negative'
                confidence = negative_score
        
        # Adjust confidence based on sentence length and match count
        if total_matches > 0:
            # More matches = higher confidence
            confidence = min(confidence + (total_matches * 0.1), 0.95)
        
        return {
            'label': sentiment,
            'confidence': float(confidence),
            'scores': {
                'positive_matches': positive_matches,
                'negative_matches': negative_matches,
                'positive_score': positive_matches / max(total_matches, 1),
                'negative_score': negative_matches / max(total_matches, 1)
            },
            'method': 'keyword_heuristic'
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
        labels: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculate aggregate sentiment for each cluster.
        
        Args:
            sentences: List of all sentences
            embeddings: Sentence embeddings (unused in this implementation)
            labels: Cluster labels (-1 for noise)
            
        Returns:
            Dictionary mapping cluster label to sentiment analysis
        """
        cluster_sentiments = {}
        unique_labels = np.unique(labels)
        
        logger.debug(f"Aggregating sentiment for {len(unique_labels)} clusters")
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            
            cluster_indices = np.where(labels == label)[0]
            cluster_sentences = [sentences[i] for i in cluster_indices]
            
            if not cluster_sentences:
                continue
            
            # Analyze each sentence in cluster
            sentiment_results = self.analyze_batch(cluster_sentences)
            
            # Aggregate sentiment
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            confidences = []
            
            for result in sentiment_results:
                sentiment_counts[result['label']] += 1
                confidences.append(result['confidence'])
            
            # Determine dominant sentiment
            total = len(sentiment_results)
            if total == 0:
                continue
            
            # Calculate proportions
            proportions = {k: v/total for k, v in sentiment_counts.items()}
            
            # Determine overall sentiment
            max_prop = max(proportions.values())
            dominant_sentiments = [k for k, v in proportions.items() if v == max_prop]
            
            if len(dominant_sentiments) == 1:
                overall_sentiment = dominant_sentiments[0]
            else:
                # Tie - use confidence-weighted decision
                sentiment_scores = {}
                for sentiment in ['positive', 'negative', 'neutral']:
                    # Average confidence for this sentiment
                    sentiment_confidences = [
                        result['confidence'] 
                        for result in sentiment_results 
                        if result['label'] == sentiment
                    ]
                    sentiment_scores[sentiment] = (
                        np.mean(sentiment_confidences) if sentiment_confidences else 0
                    )
                
                overall_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Calculate sentiment strength (how strongly positive/negative)
            if overall_sentiment == 'positive':
                sentiment_strength = proportions.get('positive', 0)
            elif overall_sentiment == 'negative':
                sentiment_strength = proportions.get('negative', 0)
            else:
                sentiment_strength = proportions.get('neutral', 0)
            
            cluster_sentiments[label] = {
                'sentiment': overall_sentiment,
                'confidence': float(avg_confidence),
                'strength': float(sentiment_strength),
                'proportions': proportions,
                'counts': sentiment_counts,
                'total_sentences': total
            }
        
        return cluster_sentiments
    
    def add_custom_keywords(
        self, 
        positive_keywords: List[str], 
        negative_keywords: List[str]
    ) -> None:
        """
        Add custom keywords to the analyzer.
        
        Args:
            positive_keywords: Additional positive keywords
            negative_keywords: Additional negative keywords
        """
        self.positive_keywords.extend(positive_keywords)
        self.negative_keywords.extend(negative_keywords)
        
        # Recompile patterns
        self.positive_pattern = re.compile(
            r'\b(' + '|'.join(self.positive_keywords) + r')\b',
            re.IGNORECASE
        )
        self.negative_pattern = re.compile(
            r'\b(' + '|'.join(self.negative_keywords) + r')\b',
            re.IGNORECASE
        )
        
        logger.debug(
            f"Added {len(positive_keywords)} positive and "
            f"{len(negative_keywords)} negative keywords"
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