import unittest
from unittest.mock import MagicMock, patch, Mock
import numpy as np
from src.app.pipeline.orchestrator import PipelineOrchestrator

class TestPipelineOrchestrator(unittest.TestCase):
    def setUp(self):
        # Create mock services for dependency injection
        self.mock_embedding_service = Mock()
        self.mock_clustering_service = Mock()
        self.mock_sentiment_service = Mock()
        
        # Patch other dependencies that are still instantiated directly
        self.patcher_preproc = patch('src.app.pipeline.orchestrator.TextPreprocessor')
        self.patcher_insights = patch('src.app.pipeline.orchestrator.InsightGenerator')
        self.patcher_compare = patch('src.app.pipeline.orchestrator.ComparisonAnalyzer')
        
        self.MockTextPreprocessor = self.patcher_preproc.start()
        self.MockInsightGenerator = self.patcher_insights.start()
        self.MockComparisonAnalyzer = self.patcher_compare.start()
        
        # Create orchestrator with mocked services
        self.orchestrator = PipelineOrchestrator(
            embedding_service=self.mock_embedding_service,
            clustering_service=self.mock_clustering_service,
            sentiment_service=self.mock_sentiment_service
        )
        
    def tearDown(self):
        self.patcher_preproc.stop()
        self.patcher_insights.stop()
        self.patcher_compare.stop()

    def test_process_standalone_parallel_execution(self):
        # Setup mocks
        sentences = [{'sentence': 'test sentence', 'id': '1'}]
        preprocessed = ['test sentence']
        embeddings = np.array([[0.1, 0.2]])
        labels = np.array([0])
        sentiment_results = [{'label': 'positive', 'scores': {'compound': 0.8}}]
        cluster_sentiments = {
            0: {'sentiment': 'positive', 'confidence': 0.9}
        }
        
        self.orchestrator.preprocessor.preprocess_batch.return_value = preprocessed
        self.mock_embedding_service.embed.return_value = embeddings
        self.mock_sentiment_service.analyze_batch.return_value = sentiment_results
        self.mock_clustering_service.cluster.return_value = labels
        self.mock_sentiment_service.get_cluster_sentiment.return_value = cluster_sentiments
        self.orchestrator.insight_generator.generate_cluster_insights.return_value = {'title': 'Test', 'insights': ['Insight']}

        # Call the method
        with patch('concurrent.futures.ThreadPoolExecutor') as MockExecutor:
            mock_executor_instance = MockExecutor.return_value
            mock_executor_instance.__enter__.return_value = mock_executor_instance
            
            future_sentiment = MagicMock()
            future_sentiment.result.return_value = sentiment_results
            
            mock_executor_instance.submit.return_value = future_sentiment
            
            result = self.orchestrator.process_standalone(sentences, "job-123")
            
            # Verify ThreadPoolExecutor was used for sentiment
            MockExecutor.assert_called()
            mock_executor_instance.submit.assert_called_with(self.mock_sentiment_service.analyze_batch, preprocessed)
            
            # Verify direct calls for others
            self.mock_embedding_service.embed.assert_called_with(preprocessed)
            self.mock_clustering_service.cluster.assert_called_with(embeddings)
            
            # Verify results
            self.assertEqual(len(result['clusters']), 1)
            self.assertEqual(result['clusters'][0]['title'], 'Test')

    def test_process_comparison_calls_process_dataset(self):
        # This indirectly tests _process_dataset which also uses parallel execution
        baseline = [{'sentence': 'b1', 'id': 'b1'}]
        comparison = [{'sentence': 'c1', 'id': 'c1'}]
        
        # Mock internal _process_dataset to simplify
        with patch.object(self.orchestrator, '_process_dataset') as mock_process_dataset:
            mock_process_dataset.return_value = {'clusters': []}
            self.orchestrator.comparison_analyzer.compare_clusters.return_value = {}
            
            self.orchestrator.process_comparison(baseline, comparison, "job-456")
            
            # Verify it's called twice (once for baseline, once for comparison)
            self.assertEqual(mock_process_dataset.call_count, 2)

if __name__ == '__main__':
    unittest.main()
