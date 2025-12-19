import json
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.app.pipeline.orchestrator import PipelineOrchestrator

class TestRealDataPipeline:
    """
    Integration tests using real data files from the data/ directory.
    Mocks the embedding model to avoid downloading/running heavy models,
    but executes the rest of the pipeline logic.
    """

    def setup_method(self):
        self.orchestrator = PipelineOrchestrator()
        self.data_dir = os.path.join(os.path.dirname(__file__), '../../data')

    def load_json(self, filename):
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            return json.load(f)

    def test_process_standalone_real_data(self):
        """Test processing data/input_example.json as a standalone request."""
        data = self.load_json('input_example.json')
        sentences = data['baseline']
        
        # Ensure we have data
        assert len(sentences) > 0
        
        # Mock embedding model to return random embeddings
        # shape: (n_sentences, 384) - 384 is dimension of all-MiniLM-L6-v2
        with patch.object(self.orchestrator.embedding_model, 'embed') as mock_embed:
            mock_embed.return_value = np.random.rand(len(sentences), 384)
            
            # Run the pipeline
            result = self.orchestrator.process_standalone(
                sentences=sentences,
                job_id="test-real-standalone"
            )
            
            # Verify output structure
            assert 'clusters' in result
            assert isinstance(result['clusters'], list)
            
            # Check cluster structure
            if len(result['clusters']) > 0:
                cluster = result['clusters'][0]
                assert 'title' in cluster
                assert 'sentiment' in cluster
                assert cluster['sentiment'] in ['positive', 'negative', 'neutral']
                assert 'sentences' in cluster
                assert isinstance(cluster['sentences'], list)
                assert 'keyInsights' in cluster
                assert isinstance(cluster['keyInsights'], list)

    def test_process_comparison_real_data(self):
        """Test processing data/input_comparison_example.json as a comparison request."""
        data = self.load_json('input_comparison_example.json')
        baseline = data['baseline']
        comparison = data['comparison']
        
        # Ensure we have data
        assert len(baseline) > 0
        assert len(comparison) > 0
        
        # Mock embedding model
        # We need to mock it to handle calls for both datasets
        # The orchestrator calls embed() separately for baseline and comparison
        with patch.object(self.orchestrator.embedding_model, 'embed') as mock_embed:
            def side_effect(sentences):
                return np.random.rand(len(sentences), 384)
            
            mock_embed.side_effect = side_effect
            
            # Run the pipeline
            result = self.orchestrator.process_comparison(
                baseline=baseline,
                comparison=comparison,
                job_id="test-real-comparison"
            )
            
            # Verify output structure
            assert 'clusters' in result
            assert isinstance(result['clusters'], list)
            assert 'metadata' in result
            
            # Check metadata
            metadata = result['metadata']
            assert 'processingTimeMs' in metadata
            assert 'similarityScore' in metadata
            assert metadata['totalBaselineSentences'] == len(baseline)
            assert metadata['totalComparisonSentences'] == len(comparison)
            
            # Check cluster structure
            if len(result['clusters']) > 0:
                cluster = result['clusters'][0]
                assert 'title' in cluster
                assert 'sentiment' in cluster
                assert 'baselineSentences' in cluster
                assert isinstance(cluster['baselineSentences'], list)
                assert 'comparisonSentences' in cluster
                assert isinstance(cluster['comparisonSentences'], list)
                assert 'keySimilarities' in cluster
                assert isinstance(cluster['keySimilarities'], list)
                assert 'keyDifferences' in cluster
                assert isinstance(cluster['keyDifferences'], list)
