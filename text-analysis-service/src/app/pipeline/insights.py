"""
Insight generation module.

Generate insights:
- Cluster title
- 2-3 bullet-point insights
- Use deterministic or template-based logic
- Avoid LLM calls unless clearly marked as optional
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter
import re

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class InsightGenerator:
    """
    Generate insights from clustered sentences.
    
    Uses deterministic or template-based logic to create:
    - Cluster titles
    - 2-3 bullet-point insights
    - Representative sentences
    - Key terms
    
    Avoids LLM calls for deterministic behavior.
    """
    
    def __init__(
        self,
        max_title_length: int = 100,
        max_insights_per_cluster: int = 3,
        min_sentence_length: int = 10
    ):
        """
        Initialize the insight generator.
        
        Args:
            max_title_length: Maximum length for cluster titles
            max_insights_per_cluster: Maximum insights per cluster
            min_sentence_length: Minimum sentence length to consider
        """
        self.max_title_length = max_title_length
        self.max_insights_per_cluster = max_insights_per_cluster
        self.min_sentence_length = min_sentence_length
        
        # Template-based insight patterns
        self.insight_templates = {
            'size': [
                "This cluster contains {count} related comments.",
                "{count} respondents mentioned this topic.",
                "A group of {count} similar feedback points."
            ],
            'sentiment_positive': [
                "Overall positive sentiment ({confidence:.0%} confidence).",
                "Respondents expressed satisfaction with this topic.",
                "Positive feedback dominates this cluster."
            ],
            'sentiment_negative': [
                "Overall negative sentiment ({confidence:.0%} confidence).",
                "Concerns were raised about this topic.",
                "Negative feedback is prominent in this cluster."
            ],
            'sentiment_neutral': [
                "Neutral sentiment observed ({confidence:.0%} confidence).",
                "Mixed or neutral opinions on this topic.",
                "No strong sentiment direction detected."
            ],
            'diversity': [
                "Moderate diversity of opinions within the cluster.",
                "Relatively homogeneous perspectives.",
                "Varied expressions of similar ideas."
            ]
        }
        
        logger.debug("InsightGenerator initialized with template-based logic")
    
    def generate_cluster_insights(
        self,
        cluster_id: int,
        sentences: List[str],
        embeddings: np.ndarray,
        sentiment_info: Dict[str, Any],
        preprocessor: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Generate insights for a single cluster.
        
        Args:
            cluster_id: Cluster identifier
            sentences: Sentences in the cluster
            embeddings: Sentence embeddings
            sentiment_info: Sentiment analysis results for the cluster
            preprocessor: TextPreprocessor instance
            
        Returns:
            Dictionary with cluster insights, or None if cluster is empty
        """
        if not sentences:
            logger.debug(f"Empty cluster {cluster_id}, skipping insight generation")
            return None
        
        logger.debug(f"Generating insights for cluster {cluster_id} with {len(sentences)} sentences")
        
        try:
            # Extract key terms using preprocessor
            all_text = ' '.join(sentences)
            key_terms = self._extract_key_terms(all_text, preprocessor)
            
            # Generate cluster title
            title = self._generate_cluster_title(sentences, key_terms)
            
            # Get representative sentences (closest to centroid)
            representative_sentences = self._get_representative_sentences(
                sentences, embeddings
            )
            
            # Calculate sentence diversity
            diversity_score = self._calculate_diversity(embeddings)
            
            # Generate insights
            insights = self._generate_insights(
                sentences=sentences,
                sentiment_info=sentiment_info,
                diversity_score=diversity_score,
                key_terms=key_terms
            )
            
            # Generate summary
            summary = self._generate_summary(sentences, sentiment_info)
            
            return {
                'cluster_id': cluster_id,
                'title': title,
                'representative_sentences': representative_sentences,
                'sentiment': sentiment_info.get('sentiment', 'neutral'),
                'sentiment_confidence': sentiment_info.get('confidence', 0.0),
                'sentiment_strength': sentiment_info.get('strength', 0.0),
                'key_terms': key_terms,
                'insights': insights,
                'sentence_count': len(sentences),
                'diversity_score': diversity_score,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Failed to generate insights for cluster {cluster_id}: {str(e)}")
            # Return minimal insights
            return {
                'cluster_id': cluster_id,
                'title': f"Cluster {cluster_id}",
                'representative_sentences': sentences[:3] if sentences else [],
                'sentiment': sentiment_info.get('sentiment', 'neutral'),
                'sentiment_confidence': sentiment_info.get('confidence', 0.0),
                'sentiment_strength': sentiment_info.get('strength', 0.0),
                'key_terms': [],
                'insights': ["Error generating detailed insights"],
                'sentence_count': len(sentences),
                'diversity_score': 0.0,
                'summary': f"Cluster with {len(sentences)} sentences"
            }
    
    def _extract_key_terms(
        self, 
        text: str, 
        preprocessor: Any
    ) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text: Text to extract terms from
            preprocessor: TextPreprocessor instance
            
        Returns:
            List of key terms
        """
        # TODO: Use preprocessor.extract_key_terms when available
        # For now, use simple heuristic
        
        # Remove common stop words and short words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Remove very common words
        common_words = {'the', 'and', 'that', 'for', 'with', 'this', 'have', 'from'}
        filtered_counts = {
            word: count 
            for word, count in word_counts.items() 
            if word not in common_words and count >= 2
        }
        
        # Get top terms
        top_terms = [
            word for word, _ in 
            sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return top_terms
    
    def _generate_cluster_title(
        self, 
        sentences: List[str], 
        key_terms: List[str]
    ) -> str:
        """
        Generate a descriptive title for the cluster.
        
        Args:
            sentences: Sentences in the cluster
            key_terms: Key terms extracted from the cluster
            
        Returns:
            Cluster title
        """
        if not sentences:
            return "Empty Cluster"
        
        # Try to create title from key terms
        if key_terms:
            # Use the most frequent key terms
            title_terms = key_terms[:3]
            title = " ".join(title_terms).title()
            
            # Truncate if too long
            if len(title) > self.max_title_length:
                title = title[:self.max_title_length-3] + "..."
            
            return title
        
        # Fallback: use first sentence truncated
        first_sentence = sentences[0]
        if len(first_sentence) > self.max_title_length:
            return first_sentence[:self.max_title_length-3] + "..."
        
        return first_sentence
    
    def _get_representative_sentences(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[str]:
        """
        Get representative sentences (closest to centroid).
        
        Args:
            sentences: Sentences in the cluster
            embeddings: Sentence embeddings
            
        Returns:
            List of representative sentences
        """
        if len(sentences) <= 3:
            return sentences
        
        try:
            # Calculate centroid
            centroid = np.mean(embeddings, axis=0)
            
            # Calculate distances to centroid
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            
            # Get indices of closest sentences
            closest_indices = np.argsort(distances)[:3]
            
            return [sentences[i] for i in closest_indices]
            
        except Exception as e:
            logger.warning(f"Failed to calculate representative sentences: {str(e)}")
            return sentences[:3]
    
    def _calculate_diversity(self, embeddings: np.ndarray) -> float:
        """
        Calculate diversity score based on embedding variance.
        
        Args:
            embeddings: Sentence embeddings
            
        Returns:
            Diversity score (0-1)
        """
        if len(embeddings) <= 1:
            return 0.0
        
        try:
            # Calculate pairwise cosine distances
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(embeddings)
            
            # Diversity is average pairwise distance
            diversity = np.mean(distances)
            
            # Normalize to 0-1 range
            return float(np.clip(diversity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Failed to calculate diversity: {str(e)}")
            return 0.5  # Default moderate diversity
    
    def _generate_insights(
        self,
        sentences: List[str],
        sentiment_info: Dict[str, Any],
        diversity_score: float,
        key_terms: List[str]
    ) -> List[str]:
        """
        Generate 2-3 bullet-point insights for the cluster.
        
        Args:
            sentences: Sentences in the cluster
            sentiment_info: Sentiment analysis results
            diversity_score: Diversity score
            key_terms: Key terms
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Insight 1: Cluster size
        count = len(sentences)
        size_template = np.random.choice(self.insight_templates['size'])
        insights.append(size_template.format(count=count))
        
        # Insight 2: Sentiment
        sentiment = sentiment_info.get('sentiment', 'neutral')
        confidence = sentiment_info.get('confidence', 0.5)
        
        if sentiment == 'positive':
            sentiment_template = np.random.choice(self.insight_templates['sentiment_positive'])
        elif sentiment == 'negative':
            sentiment_template = np.random.choice(self.insight_templates['sentiment_negative'])
        else:
            sentiment_template = np.random.choice(self.insight_templates['sentiment_neutral'])
        
        insights.append(sentiment_template.format(confidence=confidence))
        
        # Insight 3: Key terms or diversity
        if key_terms and len(key_terms) >= 2:
            key_terms_str = ", ".join(key_terms[:3])
            insights.append(f"Key topics: {key_terms_str}")
        else:
            # Use diversity insight
            if diversity_score > 0.7:
                insights.append("High diversity of expressions within this topic.")
            elif diversity_score < 0.3:
                insights.append("Consistent phrasing and terminology used.")
            else:
                diversity_template = np.random.choice(self.insight_templates['diversity'])
                insights.append(diversity_template)
        
        # Limit to max insights
        return insights[:self.max_insights_per_cluster]
    
    def _generate_summary(
        self, 
        sentences: List[str], 
        sentiment_info: Dict[str, Any]
    ) -> str:
        """
        Generate a natural language summary of the cluster.
        
        Args:
            sentences: Sentences in the cluster
            sentiment_info: Sentiment analysis results
            
        Returns:
            Summary string
        """
        count = len(sentences)
        sentiment = sentiment_info.get('sentiment', 'neutral')
        
        if sentiment == 'positive':
            sentiment_phrase = 'positive feedback'
        elif sentiment == 'negative':
            sentiment_phrase = 'concerns'
        else:
            sentiment_phrase = 'feedback'
        
        # Create summary based on cluster size and sentiment
        if count == 1:
            return f"A single {sentiment} comment."
        elif count <= 3:
            return f"A few {sentiment_phrase} on this topic."
        elif count <= 10:
            return f"Multiple {sentiment_phrase} from {count} respondents."
        else:
            return f"Strong cluster with {count} {sentiment_phrase}."
    
    def generate_overall_summary(
        self, 
        clusters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate overall summary across all clusters.
        
        Args:
            clusters: List of cluster insights
            
        Returns:
            Dictionary with overall summary
        """
        if not clusters:
            return {
                'total_clusters': 0,
                'total_sentences': 0,
                'summary': "No clusters found in the analysis."
            }
        
        total_sentences = sum(c.get('sentence_count', 0) for c in clusters)
        sentiment_distribution = Counter()
        
        for cluster in clusters:
            sentiment = cluster.get('sentiment', 'neutral')
            count = cluster.get('sentence_count', 0)
            sentiment_distribution[sentiment] += count
        
        # Calculate percentages
        sentiment_percentages = {
            sentiment: count/total_sentences 
            for sentiment, count in sentiment_distribution.items()
        }
        
        # Identify dominant sentiment
        if sentiment_distribution:
            dominant = max(sentiment_distribution.items(), key=lambda x: x[1])[0]
        else:
            dominant = 'neutral'
        
        # Find largest cluster
        if clusters:
            largest_cluster = max(clusters, key=lambda x: x.get('sentence_count', 0))
            largest_title = largest_cluster.get('title', 'Unknown')
            largest_size = largest_cluster.get('sentence_count', 0)
            largest_sentiment = largest_cluster.get('sentiment', 'neutral')
        else:
            largest_title = "None"
            largest_size = 0
            largest_sentiment = "neutral"
        
        # Generate overall summary text
        if total_sentences == 0:
            summary_text = "No sentences were analyzed."
        elif len(clusters) == 1:
            summary_text = f"Analysis identified 1 main theme with {total_sentences} comments."
        else:
            summary_text = (
                f"Analysis identified {len(clusters)} distinct themes "
                f"from {total_sentences} total comments. "
                f"The dominant sentiment is {dominant}."
            )
        
        return {
            'total_clusters': len(clusters),
            'total_sentences': total_sentences,
            'sentiment_distribution': dict(sentiment_distribution),
            'sentiment_percentages': sentiment_percentages,
            'dominant_sentiment': dominant,
            'largest_cluster': {
                'title': largest_title,
                'size': largest_size,
                'sentiment': largest_sentiment
            },
            'average_cluster_size': total_sentences / len(clusters) if clusters else 0,
            'summary': summary_text
        }


# Singleton instance for easy import
_default_insight_generator: Optional[InsightGenerator] = None


def get_insight_generator() -> InsightGenerator:
    """
    Get or create the default insight generator instance.
    
    Returns:
        Shared InsightGenerator instance
    """
    global _default_insight_generator
    if _default_insight_generator is None:
        _default_insight_generator = InsightGenerator()
        logger.info("Created default insight generator instance")
    
    return _default_insight_generator