"""
Integration tests for comparative text analysis pipeline.

Tests the full end-to-end workflow for comparing two sets of sentences,
validating that comparison analysis works correctly and produces expected outputs.
"""

import pytest
import numpy as np
from unittest.mock import patch

from src.app.pipeline.orchestrator import PipelineOrchestrator


class TestComparisonPipelineIntegration:
    """Integration tests for comparative analysis pipeline."""
    
    def test_process_comparison_basic(self):
        """Test basic comparative processing with mock components."""
        orchestrator = PipelineOrchestrator()
        
        # Mock all components
        with patch.object(orchestrator.preprocessor, 'preprocess_batch') as mock_preprocess:
            with patch.object(orchestrator.embedding_model, 'embed') as mock_embed:
                with patch.object(orchestrator.cluster_analyzer, 'cluster') as mock_cluster:
                    with patch.object(orchestrator.sentiment_analyzer, 'get_cluster_sentiment') as mock_sentiment:
                        with patch.object(orchestrator.insight_generator, 'generate_cluster_insights') as mock_insights:
                            with patch.object(orchestrator.insight_generator, 'generate_overall_summary') as mock_summary:
                                with patch.object(orchestrator.comparison_analyzer, 'compare_clusters') as mock_compare:
                                    
                                    # Setup mock returns
                                    mock_preprocess.return_value = ["preprocessed1", "preprocessed2"]
                                    mock_embed.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
                                    mock_cluster.return_value = np.array([0, 0])
                                    mock_sentiment.return_value = {0: {'sentiment': 'positive', 'confidence': 0.8}}
                                    mock_insights.return_value = {
                                        'cluster_id': 0,
                                        'title': 'Test Cluster',
                                        'representative_sentences': ['preprocessed1'],
                                        'sentiment': 'positive',
                                        'sentiment_confidence': 0.8,
                                        'key_terms': ['test'],
                                        'sentence_count': 2,
                                        'diversity_score': 0.1,
                                        'patterns': [],
                                        'summary': 'Test summary'
                                    }
                                    mock_summary.return_value = {
                                        'total_clusters': 1,
                                        'total_sentences': 2,
                                        'sentiment_distribution': {'positive': 2},
                                        'sentiment_percentages': {'positive': 1.0},
                                        'dominant_sentiment': 'positive',
                                        'largest_cluster': {
                                            'title': 'Test Cluster',
                                            'size': 2,
                                            'sentiment': 'positive'
                                        },
                                        'average_cluster_size': 2.0
                                    }
                                    mock_compare.return_value = {
                                        'similarities': [],
                                        'differences': {
                                            'baseline_unique': [],
                                            'comparison_unique': []
                                        },
                                        'similarity_score': 0.5,
                                        'summary': 'Datasets are moderately similar',
                                        'baseline_cluster_count': 1,
                                        'comparison_cluster_count': 1,
                                        'similarity_pair_count': 0
                                    }
                                    
                                    # Process comparative analysis
                                    result = orchestrator.process_comparison(
                                        baseline=[{"sentence": "baseline1", "id": "b1"}, {"sentence": "baseline2", "id": "b2"}],
                                        comparison=[{"sentence": "comparison1", "id": "c1"}, {"sentence": "comparison2", "id": "c2"}],
                                        job_id="test-job-comparison-123"
                                    )
                                    
                                    # Verify structure (new format)
                                    assert 'baseline' in result
                                    assert 'comparison' in result
                                    assert 'comparison_analysis' in result
                                    assert 'processing_metadata' in result
                                    assert 'clusters' in result['baseline']
                                    assert 'summary' in result['baseline']
                                    assert 'clusters' in result['comparison']
                                    assert 'summary' in result['comparison']
                                    assert 'similarities' in result['comparison_analysis']
                                    assert 'differences' in result['comparison_analysis']
                                    assert 'similarity_score' in result['comparison_analysis']
    
    def test_process_comparison_empty_datasets(self):
        """Test comparative processing with empty datasets."""
        orchestrator = PipelineOrchestrator()
        
        # Test with empty baseline
        result = orchestrator.process_comparison(
            baseline=[],
            comparison=[{"sentence": "comparison", "id": "c1"}],
            job_id="test-job-empty-baseline"
        )
        
        assert 'baseline' in result
        assert 'comparison' in result
        assert result['processing_metadata']['baseline_sentence_count'] == 0
        
        # Test with empty comparison
        result = orchestrator.process_comparison(
            baseline=[{"sentence": "baseline", "id": "b1"}],
            comparison=[],
            job_id="test-job-empty-comparison"
        )
        
        assert result['processing_metadata']['comparison_sentence_count'] == 0
        
        # Test with both empty
        result = orchestrator.process_comparison(
            baseline=[],
            comparison=[],
            job_id="test-job-both-empty"
        )
        
        assert result['processing_metadata']['baseline_sentence_count'] == 0
        assert result['processing_metadata']['comparison_sentence_count'] == 0
    
    def test_process_comparison_error_handling(self):
        """Test error handling in comparative processing."""
        orchestrator = PipelineOrchestrator()
        
        # Mock embedding to raise exception
        with patch.object(orchestrator.embedding_model, 'embed', side_effect=Exception("Embedding failed")):
            
            # Should propagate exception
            with pytest.raises(Exception, match="Embedding failed"):
                orchestrator.process_comparison(
                    baseline=[{"sentence": "test", "id": "t1"}],
                    comparison=[{"sentence": "test2", "id": "t2"}],
                    job_id="test-job-error"
                )
    
    def test_comparison_output_schema_compatibility(self):
        """Test that comparison output is compatible with ComparisonOutput schema."""
        orchestrator = PipelineOrchestrator()
        
        # Mock all components to produce schema-compatible output
        with patch.object(orchestrator.preprocessor, 'preprocess_batch') as mock_preprocess:
            with patch.object(orchestrator.embedding_model, 'embed') as mock_embed:
                with patch.object(orchestrator.cluster_analyzer, 'cluster') as mock_cluster:
                    with patch.object(orchestrator.sentiment_analyzer, 'get_cluster_sentiment') as mock_sentiment:
                        with patch.object(orchestrator.insight_generator, 'generate_cluster_insights') as mock_insights:
                            with patch.object(orchestrator.insight_generator, 'generate_overall_summary') as mock_summary:
                                with patch.object(orchestrator.comparison_analyzer, 'compare_clusters') as mock_compare:
                                    
                                    # Setup mock returns
                                    mock_preprocess.return_value = ["preprocessed1"]
                                    mock_embed.return_value = np.array([[0.1, 0.2]])
                                    mock_cluster.return_value = np.array([0])
                                    mock_sentiment.return_value = {0: {'sentiment': 'positive', 'confidence': 0.8}}
                                    mock_insights.return_value = {
                                        'cluster_id': 0,
                                        'title': 'Test Cluster',
                                        'representative_sentences': ['preprocessed1'],
                                        'sentiment': 'positive',
                                        'sentiment_confidence': 0.8,
                                        'key_terms': ['test'],
                                        'sentence_count': 1,
                                        'diversity_score': 0.1,
                                        'patterns': [],
                                        'summary': 'Test summary'
                                    }
                                    mock_summary.return_value = {
                                        'total_clusters': 1,
                                        'total_sentences': 1,
                                        'sentiment_distribution': {'positive': 1},
                                        'sentiment_percentages': {'positive': 1.0},
                                        'dominant_sentiment': 'positive',
                                        'largest_cluster': {
                                            'title': 'Test Cluster',
                                            'size': 1,
                                            'sentiment': 'positive'
                                        },
                                        'average_cluster_size': 1.0
                                    }
                                    mock_compare.return_value = {
                                        'similarities': [],
                                        'differences': {
                                            'baseline_unique': [],
                                            'comparison_unique': []
                                        },
                                        'similarity_score': 0.5,
                                        'summary': 'Datasets are moderately similar',
                                        'baseline_cluster_count': 1,
                                        'comparison_cluster_count': 1,
                                        'similarity_pair_count': 0
                                    }
                                    
                                    # Process and get result
                                    result = orchestrator.process_comparison(
                                        baseline=[{"sentence": "schema test", "id": "st1"}],
                                        comparison=[{"sentence": "schema test2", "id": "st2"}],
                                        job_id="test-job-schema-comparison"
                                    )
                                    
                                    # Convert to ComparisonOutput schema
                                    assert isinstance(result, dict)
                                    assert 'baseline' in result
                                    assert 'comparison' in result
                                    assert 'comparison_analysis' in result
                                    assert 'processing_metadata' in result
                                    assert 'clusters' in result['baseline']
                                    assert 'summary' in result['baseline']
                                    assert 'clusters' in result['comparison']
                                    assert 'summary' in result['comparison']
                                    assert 'similarities' in result['comparison_analysis']
                                    assert 'differences' in result['comparison_analysis']
                                    assert 'similarity_score' in result['comparison_analysis']
    
    def test_comparison_analyzer_integration(self):
        """Test the comparison analyzer directly with mock clusters."""
        from src.app.pipeline.comparison import ComparisonAnalyzer
        
        analyzer = ComparisonAnalyzer(similarity_threshold=0.6)
        
        # Create mock clusters
        baseline_clusters = [
            {
                'cluster_id': 0,
                'title': 'User Interface Issues',
                'sentiment': 'negative',
                'sentence_count': 5,
                'key_terms': ['interface', 'slow', 'buggy'],
                'representative_sentences': ['The interface is too slow']
            }
        ]
        
        comparison_clusters = [
            {
                'cluster_id': 0,
                'title': 'UI Problems',
                'sentiment': 'negative',
                'sentence_count': 4,
                'key_terms': ['interface', 'problems', 'slow'],
                'representative_sentences': ['UI has problems']
            }
        ]
        
        # Compare clusters
        result = analyzer.compare_clusters(
            baseline_clusters=baseline_clusters,
            comparison_clusters=comparison_clusters
        )
        
        # Verify structure
        assert 'similarities' in result
        assert 'differences' in result
        assert 'similarity_score' in result
        assert 'summary' in result
        
        # Similarity score should be between 0 and 1
        assert 0.0 <= result['similarity_score'] <= 1.0
        
        # Should have baseline and comparison cluster counts
        assert result['baseline_cluster_count'] == 1
        assert result['comparison_cluster_count'] == 1


# TODO: In production, add more comprehensive tests:
# - Test with actual sentence data (not mocked)
# - Test similarity threshold edge cases
# - Test performance with large datasets
# - Test schema validation with Pydantic
# - Test error handling for malformed inputs
# - Test deterministic behavior across runs
