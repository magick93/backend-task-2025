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
import concurrent.futures

from .preprocessing import TextPreprocessor
from .interfaces import EmbeddingService, ClusteringService, SentimentService
from .factory import ServiceFactory
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
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        clustering_service: Optional[ClusteringService] = None,
        sentiment_service: Optional[SentimentService] = None
    ):
        """
        Initialize pipeline components with dependency injection.
        
        Args:
            embedding_service: Optional embedding service instance.
                               If None, uses ServiceFactory.get_embedding_service()
            clustering_service: Optional clustering service instance.
                               If None, uses ServiceFactory.get_clustering_service()
            sentiment_service: Optional sentiment service instance.
                               If None, uses ServiceFactory.get_sentiment_service()
        """
        self.preprocessor = TextPreprocessor()
        self.embedding_service = embedding_service or ServiceFactory.get_embedding_service()
        self.clustering_service = clustering_service or ServiceFactory.get_clustering_service()
        self.sentiment_service = sentiment_service or ServiceFactory.get_sentiment_service()
        self.insight_generator = InsightGenerator()
        self.comparison_analyzer = ComparisonAnalyzer()
        
        logger.debug("PipelineOrchestrator initialized with dependency injection")
    
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
            
            # Step 2 & 3: Run Sentiment Analysis in parallel with (Embedding + Clustering)
            logger.debug("Starting parallel execution of sentiment analysis and (embedding + clustering)")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Start sentiment analysis in background
                future_sentiment = executor.submit(self.sentiment_service.analyze_batch, preprocessed_sentences)
                
                # Run embedding and clustering in main thread (or could be another future)
                logger.debug("Generating sentence embeddings")
                embeddings = self.embedding_service.embed(preprocessed_sentences)
                
                logger.debug("Clustering sentences")
                labels = self.clustering_service.cluster(embeddings)
                
                # Get sentiment results
                sentiment_results = future_sentiment.result()
                
            logger.debug("Parallel execution completed")
            
            # Step 5: Aggregate sentiment by cluster
            logger.debug("Aggregating sentiment by cluster")
            cluster_sentiments = self.sentiment_service.get_cluster_sentiment(
                sentences=preprocessed_sentences,
                embeddings=embeddings,
                labels=labels,
                precomputed_sentiments=sentiment_results
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
                    preprocessor=self.preprocessor,
                    total_dataset_size=len(sentences)
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
            Dictionary with comparative analysis results matching required format:
            {
                "clusters": [
                    {
                        "title": str,
                        "sentiment": str,
                        "baselineSentences": List[str],
                        "comparisonSentences": List[str],
                        "keySimilarities": List[str],
                        "keyDifferences": List[str]
                    }
                ]
            }
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
            
            # Transform results to match schema
            comparison_clusters = []
            
            # 1. Add matched clusters (similar pairs)
            for sim in comparison_analysis.get('similarities', []):
                b_cluster = sim['baseline_cluster']
                c_cluster = sim['comparison_cluster']
                
                # Determine unified title and sentiment
                # Prefer title from larger cluster or baseline if similar size
                if b_cluster['size'] >= c_cluster['size']:
                    title = b_cluster['title']
                    sentiment = b_cluster['sentiment']
                else:
                    title = c_cluster['title']
                    sentiment = c_cluster['sentiment']
                    
                # Create ComparisonCluster object
                comparison_clusters.append({
                    'title': title,
                    'sentiment': sentiment,
                    'baselineSentences': b_cluster.get('sentence_ids', []),
                    'comparisonSentences': c_cluster.get('sentence_ids', []),
                    'keySimilarities': sim.get('key_similarities', []),
                    'keyDifferences': sim.get('key_differences', [])
                })
            
            # 2. Add unique baseline clusters
            differences = comparison_analysis.get('differences', {})
            for b_unique in differences.get('baseline_unique', []):
                comparison_clusters.append({
                    'title': b_unique.get('title', 'Unknown'),
                    'sentiment': b_unique.get('sentiment', 'neutral'),
                    'baselineSentences': b_unique.get('sentence_ids', []),
                    'comparisonSentences': [],
                    'keySimilarities': [],
                    'keyDifferences': [
                        "This topic appears uniquely in the baseline dataset.",
                        f"Key insights: {', '.join(b_unique.get('insights', [])[:1])}"
                    ]
                })
                
            # 3. Add unique comparison clusters
            for c_unique in differences.get('comparison_unique', []):
                comparison_clusters.append({
                    'title': c_unique.get('title', 'Unknown'),
                    'sentiment': c_unique.get('sentiment', 'neutral'),
                    'baselineSentences': [],
                    'comparisonSentences': c_unique.get('sentence_ids', []),
                    'keySimilarities': [],
                    'keyDifferences': [
                        "This topic appears uniquely in the comparison dataset.",
                        f"Key insights: {', '.join(c_unique.get('insights', [])[:1])}"
                    ]
                })
            
            processing_time_ms = timer.elapsed_ms()
            
            result = {
                'clusters': comparison_clusters,
                'metadata': {
                    'processingTimeMs': processing_time_ms,
                    'similarityScore': comparison_analysis.get('similarity_score', 0.0),
                    'totalBaselineSentences': len(baseline),
                    'totalComparisonSentences': len(comparison),
                    'summary': comparison_analysis.get('summary', '')
                }
            }
            
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
        
        # Embed and Sentiment Analysis in parallel
        # We run sentiment in background while doing embedding + clustering
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_sentiment = executor.submit(self.sentiment_service.analyze_batch, preprocessed_sentences)
            
            # Embed
            embeddings = self.embedding_service.embed(preprocessed_sentences)
            
            # Cluster
            labels = self.clustering_service.cluster(embeddings)
            
            # Wait for sentiment
            sentiment_results = future_sentiment.result()
        
        # Aggregate sentiment
        cluster_sentiments = self.sentiment_service.get_cluster_sentiment(
            sentences=preprocessed_sentences,
            embeddings=embeddings,
            labels=labels,
            precomputed_sentiments=sentiment_results
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
            
            # Ensure sentiment is valid
            sentiment = sentiment_info.get('sentiment', 'neutral')
            if sentiment not in ['positive', 'negative', 'neutral']:
                sentiment = 'neutral'
            sentiment_info['sentiment'] = sentiment
            
            insight = self.insight_generator.generate_cluster_insights(
                cluster_id=int(label),
                sentences=cluster_sentences,
                embeddings=cluster_embeddings,
                sentiment_info=sentiment_info,
                preprocessor=self.preprocessor,
                total_dataset_size=len(sentences)
            )
            
            if insight:
                # Add sentence IDs and standardized sentiment to insight for downstream use
                insight['sentence_ids'] = cluster_sentence_ids
                insight['sentiment'] = sentiment
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
