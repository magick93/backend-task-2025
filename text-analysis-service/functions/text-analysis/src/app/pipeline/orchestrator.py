"""
Pipeline orchestrator for text analysis microservice.

Coordinates the full workflow:
- Detects standalone vs comparison input
- Ensures embedding and clustering run once
- Output shape differs by input type
- Keep logic readable and explicit
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .preprocessing import TextPreprocessor
from .embedding import EmbeddingModel
from .clustering import ClusterAnalyzer
from .sentiment import SentimentAnalyzer
from .insights import InsightGenerator
from .comparison import ComparisonAnalyzer
from ..utils.logging import setup_logger
from ..utils.timing import Timer

logger = setup_logger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the complete text analysis pipeline.
    
    Responsibilities:
    1. Detect input type (standalone vs comparison)
    2. Coordinate preprocessing, embedding, clustering, sentiment, insights
    3. Ensure each step runs only once per dataset
    4. Format appropriate output for each input type
    """
    
    def __init__(self):
        """Initialize pipeline components with default configurations."""
        self.preprocessor = TextPreprocessor()
        self.embedding_model = EmbeddingModel()
        self.cluster_analyzer = ClusterAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.insight_generator = InsightGenerator()
        self.comparison_analyzer = ComparisonAnalyzer()
        
        logger.debug("PipelineOrchestrator initialized with all components")
    
    def process_standalone(
        self,
        sentences: List[Dict[str, str]],
        job_id: str
    ) -> Dict[str, Any]:
        """
        Process a standalone text analysis request.
        
        Args:
            sentences: List of sentence objects with 'sentence' (text) and 'id' fields
            job_id: Unique job identifier for logging
            
        Returns:
            Dictionary with analysis results matching required output format:
            {
                "clusters": [
                    {
                        "title": str,
                        "sentiment": "positive|negative|neutral",
                        "sentences": List[str],  # sentence IDs
                        "keyInsights": List[str]
                    }
                ]
            }
        """
        timer = Timer()
        logger.info(f"Starting standalone analysis for job {job_id}")
        
        try:
            # Extract sentence texts and IDs
            sentence_texts = [s['sentence'] for s in sentences]
            sentence_ids = [s['id'] for s in sentences]
            
            # Step 1: Preprocess text
            logger.debug(f"Preprocessing {len(sentence_texts)} sentences")
            preprocessed_sentences = self.preprocessor.preprocess_batch(sentence_texts)
            
            # Step 2: Generate embeddings
            logger.debug("Generating sentence embeddings")
            embeddings = self.embedding_model.embed(preprocessed_sentences)
            
            # Step 3: Cluster sentences
            logger.debug("Clustering sentences")
            labels = self.cluster_analyzer.cluster(embeddings)
            
            # Step 4: Analyze sentiment
            logger.debug("Analyzing sentiment")
            cluster_sentiments = self.sentiment_analyzer.get_cluster_sentiment(
                sentences=preprocessed_sentences,
                embeddings=embeddings,
                labels=labels
            )
            
            # Step 5: Generate insights for each cluster and map to required format
            logger.debug("Generating cluster insights")
            clusters = []
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                    
                cluster_indices = np.where(labels == label)[0]
                cluster_sentence_ids = [sentence_ids[i] for i in cluster_indices]
                cluster_sentences = [preprocessed_sentences[i] for i in cluster_indices]
                cluster_embeddings = embeddings[cluster_indices]
                
                sentiment_info = cluster_sentiments.get(label, {
                    'sentiment': 'neutral',
                    'confidence': 0.0
                })
                
                # Ensure sentiment is one of positive|negative|neutral (map 'mixed' to 'neutral')
                sentiment = sentiment_info.get('sentiment', 'neutral')
                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'
                
                # Generate insights using the insight generator
                insight = self.insight_generator.generate_cluster_insights(
                    cluster_id=int(label),
                    sentences=cluster_sentences,
                    embeddings=cluster_embeddings,
                    sentiment_info=sentiment_info,
                    preprocessor=self.preprocessor
                )
                
                if insight:
                    # Transform insight to required format
                    cluster_output = {
                        'title': insight.get('title', f'Cluster {label}'),
                        'sentiment': sentiment,
                        'sentences': cluster_sentence_ids,
                        'keyInsights': insight.get('insights', [])
                    }
                    clusters.append(cluster_output)
            
            # Step 6: Format results (no overall summary in required output)
            result = {
                'clusters': clusters
            }
            
            processing_time_ms = timer.elapsed_ms()
            logger.info(f"Completed standalone analysis in {processing_time_ms}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Standalone analysis failed: {str(e)}", exc_info=True)
            raise
    
    def process_comparison(
        self,
        baseline: List[Dict[str, str]],
        comparison: List[Dict[str, str]],
        job_id: str
    ) -> Dict[str, Any]:
        """
        Process a comparative text analysis request.
        
        Args:
            baseline: Baseline sentence objects with 'sentence' and 'id' fields
            comparison: Comparison sentence objects with 'sentence' and 'id' fields
            job_id: Unique job identifier for logging
            
        Returns:
            Dictionary with comparative analysis results matching required format
            (to be defined, but for now we'll adapt existing format)
        """
        timer = Timer()
        logger.info(f"Starting comparative analysis for job {job_id}")
        
        try:
            # Process baseline dataset
            logger.debug(f"Processing baseline dataset ({len(baseline)} sentences)")
            baseline_result = self._process_dataset(baseline, "baseline")
            
            # Process comparison dataset
            logger.debug(f"Processing comparison dataset ({len(comparison)} sentences)")
            comparison_result = self._process_dataset(comparison, "comparison")
            
            # Perform comparison analysis
            logger.debug("Performing comparative analysis")
            comparison_analysis = self.comparison_analyzer.compare_clusters(
                baseline_clusters=baseline_result['clusters'],
                comparison_clusters=comparison_result['clusters']
            )
            
            # Format results
            result = {
                'baseline': {
                    'clusters': baseline_result['clusters'],
                    'summary': baseline_result['summary']
                },
                'comparison': {
                    'clusters': comparison_result['clusters'],
                    'summary': comparison_result['summary']
                },
                'comparison_analysis': comparison_analysis,
                'processing_metadata': {
                    'baseline_sentence_count': len(baseline),
                    'comparison_sentence_count': len(comparison),
                    'baseline_cluster_count': len(baseline_result['clusters']),
                    'comparison_cluster_count': len(comparison_result['clusters'])
                }
            }
            
            processing_time_ms = timer.elapsed_ms()
            logger.info(f"Completed comparative analysis in {processing_time_ms}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {str(e)}", exc_info=True)
            raise
    
    def _process_dataset(
        self,
        sentences: List[Dict[str, str]],
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Process a single dataset through the pipeline.
        
        Args:
            sentences: List of sentence objects with 'sentence' and 'id' fields
            dataset_name: Name for logging purposes
            
        Returns:
            Dictionary with processed results including clusters with sentence IDs
        """
        logger.debug(f"Processing {dataset_name} dataset")
        
        # Extract texts and IDs
        sentence_texts = [s['sentence'] for s in sentences]
        sentence_ids = [s['id'] for s in sentences]
        
        # Preprocess
        preprocessed_sentences = self.preprocessor.preprocess_batch(sentence_texts)
        
        # Embed
        embeddings = self.embedding_model.embed(preprocessed_sentences)
        
        # Cluster
        labels = self.cluster_analyzer.cluster(embeddings)
        
        # Analyze sentiment
        cluster_sentiments = self.sentiment_analyzer.get_cluster_sentiment(
            sentences=preprocessed_sentences,
            embeddings=embeddings,
            labels=labels
        )
        
        # Generate insights
        clusters = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            cluster_indices = np.where(labels == label)[0]
            cluster_sentence_ids = [sentence_ids[i] for i in cluster_indices]
            cluster_sentences = [preprocessed_sentences[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            sentiment_info = cluster_sentiments.get(label, {
                'sentiment': 'neutral',
                'confidence': 0.0
            })
            
            insight = self.insight_generator.generate_cluster_insights(
                cluster_id=int(label),
                sentences=cluster_sentences,
                embeddings=cluster_embeddings,
                sentiment_info=sentiment_info,
                preprocessor=self.preprocessor
            )
            
            if insight:
                # Add sentence IDs to insight for downstream use
                insight['sentence_ids'] = cluster_sentence_ids
                clusters.append(insight)
        
        # Generate overall summary
        overall_summary = self.insight_generator.generate_overall_summary(clusters)
        
        return {
            'clusters': clusters,
            'summary': overall_summary,
            'metadata': {
                'sentence_count': len(sentences),
                'cluster_count': len(clusters),
                'noise_count': np.sum(labels == -1) if len(labels) > 0 else 0
            }
        }