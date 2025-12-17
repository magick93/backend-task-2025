"""
Integration tests for standalone text analysis pipeline.

Tests the full end-to-end workflow from input sentences to output insights,
validating that all pipeline components work together correctly.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.app.pipeline.orchestrator import PipelineOrchestrator
from src.app.api.schemas import StandaloneInput, StandaloneOutput, ClusterInsight


class TestStandalonePipelineIntegration:
    """Integration tests for standalone analysis pipeline."""
    
    def test_pipeline_orchestrator_initialization(self):
        """Test that pipeline orchestrator initializes all components."""
        orchestrator = PipelineOrchestrator()
        
        # Check that all components are initialized
        assert hasattr(orchestrator, 'preprocessor')
        assert hasattr(orchestrator, 'embedding_model')
        assert hasattr(orchestrator, 'cluster_analyzer')
        assert hasattr(orchestrator, 'sentiment_analyzer')
        assert hasattr(orchestrator, 'insight_generator')
        assert hasattr(orchestrator, 'comparison_analyzer')
    
    def test_process_standalone_basic(self):
        """Test basic standalone processing with mock components."""
        orchestrator = PipelineOrchestrator()
        
        # Mock all components
        with patch.object(orchestrator.preprocessor, 'preprocess_batch') as mock_preprocess:
            with patch.object(orchestrator.embedding_model, 'embed') as mock_embed:
                with patch.object(orchestrator.cluster_analyzer, 'cluster') as mock_cluster:
                    with patch.object(orchestrator.sentiment_analyzer, 'get_cluster_sentiment') as mock_sentiment:
                        with patch.object(orchestrator.insight_generator, 'generate_cluster_insights') as mock_insights:
                            with patch.object(orchestrator.insight_generator, 'generate_overall_summary') as mock_summary:
                                
                                # Setup mock returns
                                mock_preprocess.return_value = ["preprocessed1", "preprocessed2"]
                                mock_embed.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
                                mock_cluster.return_value = np.array([0, 0])  # Both in same cluster
                                mock_sentiment.return_value = {
                                    0: {'sentiment': 'positive', 'confidence': 0.8}
                                }
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
                                
                                # Process standalone analysis
                                result = orchestrator.process_standalone(
                                    sentences=["sentence1", "sentence2"],
                                    job_id="test-job-123"
                                )
                                
                                # Verify structure
                                assert 'clusters' in result
                                assert 'summary' in result
                                assert 'processing_metadata' in result
                                
                                # Verify mocks were called
                                mock_preprocess.assert_called_once_with(["sentence1", "sentence2"])
                                mock_embed.assert_called_once_with(["preprocessed1", "preprocessed2"])
                                mock_cluster.assert_called_once()
                                mock_sentiment.assert_called_once()
                                mock_insights.assert_called_once()
                                mock_summary.assert_called_once()
    
    def test_process_standalone_multiple_clusters(self):
        """Test standalone processing with multiple clusters."""
        orchestrator = PipelineOrchestrator()
        
        # Mock components
        with patch.object(orchestrator.preprocessor, 'preprocess_batch'):
            with patch.object(orchestrator.embedding_model, 'embed'):
                with patch.object(orchestrator.cluster_analyzer, 'cluster') as mock_cluster:
                    with patch.object(orchestrator.sentiment_analyzer, 'get_cluster_sentiment'):
                        with patch.object(orchestrator.insight_generator, 'generate_cluster_insights') as mock_insights:
                            with patch.object(orchestrator.insight_generator, 'generate_overall_summary'):
                                
                                # Setup mock returns for multiple clusters
                                mock_cluster.return_value = np.array([0, 1, 0, 2, -1])  # 3 clusters + noise
                                
                                # Mock insights to return different values based on cluster
                                def insights_side_effect(cluster_id, sentences, embeddings, sentiment_info, preprocessor):
                                    return {
                                        'cluster_id': cluster_id,
                                        'title': f'Cluster {cluster_id}',
                                        'representative_sentences': sentences[:1],
                                        'sentiment': 'positive' if cluster_id % 2 == 0 else 'negative',
                                        'sentiment_confidence': 0.7,
                                        'key_terms': ['term'],
                                        'sentence_count': len(sentences),
                                        'diversity_score': 0.2,
                                        'patterns': [],
                                        'summary': f'Summary for cluster {cluster_id}'
                                    }
                                
                                mock_insights.side_effect = insights_side_effect
                                
                                # Process
                                result = orchestrator.process_standalone(
                                    sentences=["s1", "s2", "s3", "s4", "s5"],
                                    job_id="test-job-456"
                                )
                                
                                # Should have 3 clusters (excluding noise label -1)
                                assert len(result['clusters']) == 3
                                
                                # Check processing metadata
                                metadata = result['processing_metadata']
                                assert metadata['input_sentence_count'] == 5
                                assert metadata['cluster_count'] == 3
                                assert metadata['noise_sentence_count'] == 1  # One sentence with label -1
    
    def test_process_standalone_empty_sentences(self):
        """Test standalone processing with empty sentences list."""
        orchestrator = PipelineOrchestrator()
        
        # Should handle empty list gracefully
        result = orchestrator.process_standalone(
            sentences=[],
            job_id="test-job-empty"
        )
        
        # Should still return valid structure
        assert 'clusters' in result
        assert 'summary' in result
        assert 'processing_metadata' in result
        assert result['processing_metadata']['input_sentence_count'] == 0
        assert result['processing_metadata']['cluster_count'] == 0
    
    def test_process_standalone_single_sentence(self):
        """Test standalone processing with single sentence."""
        orchestrator = PipelineOrchestrator()
        
        with patch.object(orchestrator.preprocessor, 'preprocess_batch'):
            with patch.object(orchestrator.embedding_model, 'embed'):
                with patch.object(orchestrator.cluster_analyzer, 'cluster') as mock_cluster:
                    with patch.object(orchestrator.sentiment_analyzer, 'get_cluster_sentiment'):
                        with patch.object(orchestrator.insight_generator, 'generate_cluster_insights'):
                            with patch.object(orchestrator.insight_generator, 'generate_overall_summary'):
                                
                                # Single sentence should be in its own cluster
                                mock_cluster.return_value = np.array([0])
                                
                                result = orchestrator.process_standalone(
                                    sentences=["single sentence"],
                                    job_id="test-job-single"
                                )
                                
                                assert result['processing_metadata']['input_sentence_count'] == 1
                                # Should have 1 cluster (or 0 if treated as noise)
    
    def test_process_standalone_error_handling(self):
        """Test error handling in standalone processing."""
        orchestrator = PipelineOrchestrator()
        
        # Mock embedding to raise exception
        with patch.object(orchestrator.embedding_model, 'embed', side_effect=Exception("Embedding failed")):
            
            # Should propagate exception
            with pytest.raises(Exception, match="Embedding failed"):
                orchestrator.process_standalone(
                    sentences=["test sentence"],
                    job_id="test-job-error"
                )
    
    def test_process_standalone_deterministic(self):
        """Test that standalone processing is deterministic with same inputs."""
        orchestrator1 = PipelineOrchestrator()
        orchestrator2 = PipelineOrchestrator()
        
        # Mock all components with deterministic behavior
        with patch.object(orchestrator1.preprocessor, 'preprocess_batch') as mock_preprocess1:
            with patch.object(orchestrator2.preprocessor, 'preprocess_batch') as mock_preprocess2:
                with patch.object(orchestrator1.embedding_model, 'embed') as mock_embed1:
                    with patch.object(orchestrator2.embedding_model, 'embed') as mock_embed2:
                        with patch.object(orchestrator1.cluster_analyzer, 'cluster') as mock_cluster1:
                            with patch.object(orchestrator2.cluster_analyzer, 'cluster') as mock_cluster2:
                                with patch.object(orchestrator1.sentiment_analyzer, 'get_cluster_sentiment') as mock_sentiment1:
                                    with patch.object(orchestrator2.sentiment_analyzer, 'get_cluster_sentiment') as mock_sentiment2:
                                        with patch.object(orchestrator1.insight_generator, 'generate_cluster_insights') as mock_insights1:
                                            with patch.object(orchestrator2.insight_generator, 'generate_cluster_insights') as mock_insights2:
                                                with patch.object(orchestrator1.insight_generator, 'generate_overall_summary') as mock_summary1:
                                                    with patch.object(orchestrator2.insight_generator, 'generate_overall_summary') as mock_summary2:
                                                        
                                                        # Setup identical mock returns
                                                        mock_preprocess1.return_value = mock_preprocess2.return_value = ["preprocessed"]
                                                        mock_embed1.return_value = mock_embed2.return_value = np.array([[0.1, 0.2]])
                                                        mock_cluster1.return_value = mock_cluster2.return_value = np.array([0])
                                                        mock_sentiment1.return_value = mock_sentiment2.return_value = {
                                                            0: {'sentiment': 'neutral', 'confidence': 0.5}
                                                        }
                                                        mock_insights1.return_value = mock_insights2.return_value = {
                                                            'cluster_id': 0,
                                                            'title': 'Test',
                                                            'representative_sentences': ['preprocessed'],
                                                            'sentiment': 'neutral',
                                                            'sentiment_confidence': 0.5,
                                                            'key_terms': ['test'],
                                                            'sentence_count': 1,
                                                            'diversity_score': 0.0,
                                                            'patterns': [],
                                                            'summary': 'Single comment'
                                                        }
                                                        mock_summary1.return_value = mock_summary2.return_value = {
                                                            'total_clusters': 1,
                                                            'total_sentences': 1,
                                                            'sentiment_distribution': {'neutral': 1},
                                                            'sentiment_percentages': {'neutral': 1.0},
                                                            'dominant_sentiment': 'neutral',
                                                            'largest_cluster': {
                                                                'title': 'Test',
                                                                'size': 1,
                                                                'sentiment': 'neutral'
                                                            },
                                                            'average_cluster_size': 1.0
                                                        }
                                                        
                                                        # Process with both orchestrators
                                                        result1 = orchestrator1.process_standalone(
                                                            sentences=["test"],
                                                            job_id="test-job-deterministic"
                                                        )
                                                        
                                                        result2 = orchestrator2.process_standalone(
                                                            sentences=["test"],
                                                            job_id="test-job-deterministic"
                                                        )
                                                        
                                                        # Results should be identical
                                                        assert result1 == result2
    
    def test_standalone_output_schema_compatibility(self):
        """Test that pipeline output is compatible with StandaloneOutput schema."""
        orchestrator = PipelineOrchestrator()
        
        # Mock components to produce schema-compatible output
        with patch.object(orchestrator.preprocessor, 'preprocess_batch'):
            with patch.object(orchestrator.embedding_model, 'embed'):
                with patch.object(orchestrator.cluster_analyzer, 'cluster'):
                    with patch.object(orchestrator.sentiment_analyzer, 'get_cluster_sentiment'):
                        with patch.object(orchestrator.insight_generator, 'generate_cluster_insights'):
                            with patch.object(orchestrator.insight_generator, 'generate_overall_summary'):
                                
                                # Process and get result
                                result = orchestrator.process_standalone(
                                    sentences=["schema test"],
                                    job_id="test-job-schema"
                                )
                                
                                # Convert to StandaloneOutput schema
                                # Note: The orchestrator returns a dict, not a StandaloneOutput object
                                # but the structure should be compatible
                                assert isinstance(result, dict)
                                assert 'clusters' in result
                                assert 'summary' in result
                                assert 'processing_metadata' in result
                                
                                # TODO: In production, we would validate against StandaloneOutput schema
                                # For now, just check basic structure
    
    def test_process_dataset_internal_method(self):
        """Test the internal _process_dataset method."""
        orchestrator = PipelineOrchestrator()
        
        with patch.object(orchestrator.preprocessor, 'preprocess_batch'):
            with patch.object(orchestrator.embedding_model, 'embed'):
                with patch.object(orchestrator.cluster_analyzer, 'cluster'):
                    with patch.object(orchestrator.sentiment_analyzer, 'get_cluster_sentiment'):
                        with patch.object(orchestrator.insight_generator, 'generate_cluster_insights'):
                            with patch.object(orchestrator.insight_generator, 'generate_overall_summary'):
                                
                                result = orchestrator._process_dataset(
                                    sentences=["test1", "test2"],
                                    dataset_name="test-dataset"
                                )
                                
                                assert 'clusters' in result
                                assert 'summary' in result
                                assert 'metadata' in result
                                assert result['metadata']['sentence_count'] == 2
    
    @pytest.mark.skip(reason="Requires actual ML models to be loaded")
    def test_integration_with_real_models(self):
        """TODO: Test integration with actual ML models (requires model downloads)."""
        pass
    
    @pytest.mark.skip(reason="Performance testing not implemented yet")
    def test_performance_large_dataset(self):
        """TODO: Test performance with large datasets."""
        pass


class TestStandaloneSchemaValidation:
    """Tests for schema validation in standalone analysis."""
    
    def test_standalone_input_schema_validation(self):
        """Test that StandaloneInput schema validates correctly."""
        # Valid input
        valid_input = {
            "surveyTitle": "Customer Feedback Q4 2024",
            "theme": "User Interface",
            "baseline": [
                {"sentence": "UI is intuitive", "id": "id1"},
                {"sentence": "Hard to navigate", "id": "id2"}
            ]
        }
        
        # Should create without error
        input_obj = StandaloneInput(**valid_input)
        assert input_obj.surveyTitle == "Customer Feedback Q4 2024"
        assert input_obj.theme == "User Interface"
        assert len(input_obj.baseline) == 2
        
        # Invalid input - duplicate IDs
        invalid_input = {
            "surveyTitle": "Test",
            "theme": "Test",
            "baseline": [
                {"sentence": "Test", "id": "same-id"},
                {"sentence": "Test2", "id": "same-id"}  # Duplicate ID
            ]
        }
        
        with pytest.raises(ValueError, match="All sentence IDs must be unique"):
            StandaloneInput(**invalid_input)
    
    def test_standalone_output_schema_validation(self):
        """Test that StandaloneOutput schema validates correctly."""
        # Valid output
        valid_output = {
            "clusters": [
                {
                    "title": "Money Withdrawal Issues",
                    "sentiment": "negative",
                    "sentences": ["id1", "id2"],
                    "keyInsights": ["Users report difficulty withdrawing funds"]
                }
            ],
            "metadata": {
                "processingTimeMs": 1250,
                "jobId": "job-12345"
            }
        }
        
        # Should create without error
        output_obj = StandaloneOutput(**valid_output)
        assert len(output_obj.clusters) == 1
        assert output_obj.clusters[0].title == "Money Withdrawal Issues"
        assert output_obj.clusters[0].sentiment == "negative"
        
        # Invalid output - wrong sentiment value
        invalid_output = {
            "clusters": [
                {
                    "title": "Test",
                    "sentiment": "invalid_sentiment",  # Not allowed
                    "sentences": ["id1"],
                    "keyInsights": ["Test insight"]
                }
            ]
        }
        
        with pytest.raises(ValueError, match="string does not match regex"):
            StandaloneOutput(**invalid_output)
    
    def test_schema_examples(self):
        """Test that schema examples are valid."""
        # Test StandaloneInput example
        input_example = StandaloneInput.Config.schema_extra["example"]
        input_obj = StandaloneInput(**input_example)
        assert input_obj.surveyTitle == "Robinhood App Store Reviews"
        assert input_obj.theme == "Account Management"
        assert len(input_obj.baseline) == 2
        
        # Test StandaloneOutput example
        output_example = StandaloneOutput.Config.schema_extra["example"]
        output_obj = StandaloneOutput(**output_example)
        assert len(output_obj.clusters) == 1
        assert output_obj.clusters[0].title == "Money Withdrawal Issues"
        assert output_obj.clusters[0].sentiment == "negative"
        assert len(output_obj.clusters[0].sentences) == 2
        assert len(output_obj.clusters[0].keyInsights) == 2
        
        # Test ClusterInsight example
        cluster_example = ClusterInsight.Config.schema_extra["example"]
        cluster_obj = ClusterInsight(**cluster_example)
        assert cluster_obj.title == "Money Withdrawal Issues"
        assert cluster_obj.sentiment == "negative"
        assert len(cluster_obj.sentences) == 2
        assert len(cluster_obj.keyInsights) == 2
    
    def test_end_to_end_with_fixture(self):
        """TODO: Test end-to-end with actual fixture data."""
        pass
    
    def test_output_format_matches_schema(self):
        """TODO: Test that pipeline output matches StandaloneOutput schema exactly."""
        pass