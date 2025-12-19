"""
Unit tests for the embedding module.

Tests cover:
- Embedding generation (mocked model)
- Caching behavior (hit/miss, cache eviction)
- Error handling (empty input, edge cases)
- Deterministic behavior (same input yields same embedding)
- Singleton pattern and convenience functions
"""

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.app.pipeline.embedding import (
    EmbeddingModel,
    get_embedding_model,
    embed_sentences,
)


class TestEmbeddingModel:
    """Test the EmbeddingModel class."""
    
    @pytest.fixture(autouse=True)
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer to avoid loading real model."""
        with patch('src.app.pipeline.embedding.SentenceTransformer') as MockST:
            mock_instance = MockST.return_value
            # Setup default mock behavior
            def mock_encode(sentences):
                # Return random embeddings of correct shape
                n = len(sentences) if isinstance(sentences, list) else 1
                return np.random.randn(n, 384)
            
            mock_instance.encode.side_effect = mock_encode
            yield MockST

    def test_environment_variable_set(self):
        """Test that HF_HOME environment variable is set correctly."""
        assert os.environ.get('HF_HOME') == '/tmp/huggingface'

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = EmbeddingModel()
        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.cache_size == 1000
        assert model._embedding_cache == {}
        assert model._model is None

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        model = EmbeddingModel(
            model_name="custom-model",
            cache_size=500
        )
        assert model.model_name == "custom-model"
        assert model.cache_size == 500
        assert model._embedding_cache == {}

    def test_lazy_loading(self, mock_sentence_transformer):
        """Test that model is loaded lazily."""
        model = EmbeddingModel()
        assert model._model is None
        
        # Accessing .model property should trigger load
        _ = model.model
        
        assert model._model is not None
        mock_sentence_transformer.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")

    def test_embed_empty_list(self):
        """Test embedding an empty list returns empty array."""
        model = EmbeddingModel()
        result = model.embed([])
        assert result.shape == (0,)
        assert result.size == 0

    def test_embed_single_sentence(self):
        """Test embedding a single sentence."""
        model = EmbeddingModel()
        sentences = ["Hello world"]
        
        # Override mock for this specific test to ensure normalization check passes
        # Real model returns unnormalized, but downstream might expect normalized
        # Our placeholder implementation returned normalized. 
        # The new implementation returns whatever the model returns.
        # Let's verify it calls the model and returns its output.
        
        mock_output = np.random.randn(1, 384)
        model._model = MagicMock()
        model._model.encode.return_value = mock_output
        
        result = model.embed(sentences)
        
        assert result.shape == (1, 384)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, mock_output)

    def test_embed_multiple_sentences(self):
        """Test embedding multiple sentences."""
        model = EmbeddingModel()
        sentences = [
            "Hello world",
            "This is a test sentence",
            "Another example for testing"
        ]
        
        mock_output = np.random.randn(3, 384)
        model._model = MagicMock()
        model._model.encode.return_value = mock_output
        
        result = model.embed(sentences)
        
        assert result.shape == (3, 384)
        np.testing.assert_array_equal(result, mock_output)

    def test_caching_behavior(self):
        """Test that embeddings are cached and reused."""
        model = EmbeddingModel(cache_size=10)
        sentence = "Test sentence for caching"
        
        # Prepare mock
        mock_output = np.random.randn(1, 384)
        model._model = MagicMock()
        model._model.encode.return_value = mock_output
        
        # First call - should compute
        result1 = model.embed([sentence])
        model._model.encode.assert_called_once_with([sentence])
        
        # Second call with same sentence - should use cache
        model._model.encode.reset_mock()
        result2 = model.embed([sentence])
        model._model.encode.assert_not_called()
        
        # Results should be the same
        np.testing.assert_array_equal(result1, result2)
        
        # Cache should contain the sentence
        assert sentence in model._embedding_cache

    def test_cache_mixed_sentences(self):
        """Test embedding mix of cached and uncached sentences."""
        model = EmbeddingModel()
        
        # Add first sentence to cache
        sentence1 = "Cached sentence"
        embedding1 = np.random.randn(384)
        model._embedding_cache[sentence1] = embedding1
        
        # Second sentence not in cache
        sentence2 = "Uncached sentence"
        mock_embedding = np.random.randn(1, 384)
        
        # Mock model
        model._model = MagicMock()
        model._model.encode.return_value = mock_embedding
        
        result = model.embed([sentence1, sentence2])
        
        # Check shape
        assert result.shape == (2, 384)
        
        # First row should be cached embedding
        np.testing.assert_array_equal(result[0], embedding1)
        
        # Second row should be mock embedding
        np.testing.assert_array_equal(result[1], mock_embedding[0])
        
        # Verify model was called only with uncached sentence
        model._model.encode.assert_called_once_with([sentence2])

    def test_cache_eviction(self):
        """Test that cache evicts oldest entries when full."""
        model = EmbeddingModel(cache_size=2)
        model._model = MagicMock()
        model._model.encode.side_effect = lambda s: np.random.randn(len(s), 384)
        
        # Add three sentences to cache (exceeding capacity)
        sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
        
        # Embed first two sentences
        model.embed(sentences[:2])
        assert len(model._embedding_cache) == 2
        
        # Embed third sentence - should evict oldest (Sentence 1)
        model.embed([sentences[2]])
        
        # Cache should still have size 2
        assert len(model._embedding_cache) == 2
        
        # Sentence 1 should be evicted
        assert sentences[0] not in model._embedding_cache
        assert sentences[1] in model._embedding_cache  # Second oldest
        assert sentences[2] in model._embedding_cache  # Newest

    def test_clear_cache(self):
        """Test clearing the cache."""
        model = EmbeddingModel()
        model._model = MagicMock()
        model._model.encode.return_value = np.random.randn(2, 384)
        
        # Add some sentences to cache
        sentences = ["Test 1", "Test 2"]
        model.embed(sentences)
        assert len(model._embedding_cache) == 2
        
        # Clear cache
        model.clear_cache()
        assert len(model._embedding_cache) == 0

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        model = EmbeddingModel(cache_size=100)
        model._model = MagicMock()
        model._model.encode.return_value = np.random.randn(2, 384)
        
        # Add some sentences
        sentences = ["Stat test 1", "Stat test 2"]
        model.embed(sentences)
        
        stats = model.get_cache_stats()
        assert stats['cache_size'] == 2
        assert stats['max_cache_size'] == 100

    def test_embed_uncached_returns_numpy(self):
        """Test that _embed_uncached returns numpy array even if model returns list."""
        model = EmbeddingModel()
        model._model = MagicMock()
        # Simulate model returning list of lists (common in some configurations)
        model._model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        sentences = ["A", "B"]
        result = model._embed_uncached(sentences)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_error_handling_missing_dependency(self):
        """Test error raised if sentence-transformers not available."""
        # Unpatch the mock for this test to simulate real environment check
        # But we need to simulate import error
        
        # This is tricky because the module is already imported. 
        # We'll mock the module level SentenceTransformer to be None
        
        with patch('src.app.pipeline.embedding.SentenceTransformer', None):
            model = EmbeddingModel()
            # Reset _model to force load
            model._model = None
            
            with pytest.raises(ImportError, match="sentence-transformers not installed"):
                _ = model.model


class TestSingletonAndConvenience:
    """Test the singleton pattern and convenience functions."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        # Reset before
        from src.app.pipeline import embedding
        embedding._default_embedding_model = None
        yield
        # Reset after
        embedding._default_embedding_model = None

    def test_get_embedding_model_singleton(self):
        """Test that get_embedding_model returns the same instance."""
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        
        assert model1 is model2  # Same object

    def test_embed_sentences_convenience(self):
        """Test the embed_sentences convenience function."""
        sentences = ["Convenience test 1", "Convenience test 2"]
        
        # Mock the singleton model to avoid side effects
        mock_model = MagicMock()
        mock_embeddings = np.random.randn(2, 384)
        mock_model.embed.return_value = mock_embeddings
        
        with patch('src.app.pipeline.embedding.get_embedding_model',
                   return_value=mock_model):
            result = embed_sentences(sentences)
        
        # Should call model.embed with the sentences
        mock_model.embed.assert_called_once_with(sentences)
        np.testing.assert_array_equal(result, mock_embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
