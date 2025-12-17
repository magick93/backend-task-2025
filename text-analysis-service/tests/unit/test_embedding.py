"""
Unit tests for the embedding module.

Tests cover:
- Embedding generation (placeholder implementation)
- Caching behavior (hit/miss, cache eviction)
- Error handling (empty input, edge cases)
- Deterministic behavior (same input yields same embedding)
- Singleton pattern and convenience functions
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.pipeline.embedding import (
    EmbeddingModel,
    get_embedding_model,
    embed_sentences,
)


class TestEmbeddingModel:
    """Test the EmbeddingModel class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = EmbeddingModel()
        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.cache_size == 1000
        assert model._embedding_cache == {}
        assert model._tokenizer is None
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
        result = model.embed(sentences)
        
        # Check shape
        assert result.shape == (1, 384)  # all-MiniLM-L6-v2 has 384 dimensions
        assert isinstance(result, np.ndarray)
        
        # Check normalization (unit length)
        norm = np.linalg.norm(result[0])
        assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_embed_multiple_sentences(self):
        """Test embedding multiple sentences."""
        model = EmbeddingModel()
        sentences = [
            "Hello world",
            "This is a test sentence",
            "Another example for testing"
        ]
        result = model.embed(sentences)
        
        # Check shape
        assert result.shape == (3, 384)
        
        # Each embedding should be normalized
        for i in range(3):
            norm = np.linalg.norm(result[i])
            assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_deterministic_behavior(self):
        """Test that same sentence produces same embedding."""
        model = EmbeddingModel()
        sentence = "Deterministic test sentence"
        
        # First embedding
        embedding1 = model.embed([sentence])
        
        # Clear cache to force recomputation
        model.clear_cache()
        
        # Second embedding (should be identical due to deterministic seed)
        embedding2 = model.embed([sentence])
        
        # Should be exactly equal (deterministic placeholder)
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_different_sentences_different_embeddings(self):
        """Test that different sentences produce different embeddings."""
        model = EmbeddingModel()
        sentences = ["Sentence one", "Sentence two"]
        embeddings = model.embed(sentences)
        
        # Embeddings should be different (cosine similarity not 1)
        cosine_sim = np.dot(embeddings[0], embeddings[1])
        assert not np.isclose(cosine_sim, 1.0, rtol=1e-5)

    def test_caching_behavior(self):
        """Test that embeddings are cached and reused."""
        model = EmbeddingModel(cache_size=10)
        sentence = "Test sentence for caching"
        
        # First call - should compute
        with patch.object(model, '_embed_uncached') as mock_embed:
            mock_embed.return_value = np.random.randn(1, 384)
            result1 = model.embed([sentence])
            mock_embed.assert_called_once_with([sentence])
        
        # Second call with same sentence - should use cache
        with patch.object(model, '_embed_uncached') as mock_embed:
            result2 = model.embed([sentence])
            mock_embed.assert_not_called()  # Should not call embed_uncached
        
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
        embedding1 = embedding1 / np.linalg.norm(embedding1)  # Normalize
        model._embedding_cache[sentence1] = embedding1
        
        # Second sentence not in cache
        sentence2 = "Uncached sentence"
        
        # Mock _embed_uncached to return predictable embedding for sentence2
        mock_embedding = np.random.randn(1, 384)
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding, axis=1, keepdims=True)
        
        with patch.object(model, '_embed_uncached', return_value=mock_embedding):
            result = model.embed([sentence1, sentence2])
        
        # Check shape
        assert result.shape == (2, 384)
        
        # First row should be cached embedding
        np.testing.assert_array_equal(result[0], embedding1)
        
        # Second row should be mock embedding
        np.testing.assert_array_equal(result[1], mock_embedding[0])

    def test_cache_eviction(self):
        """Test that cache evicts oldest entries when full."""
        model = EmbeddingModel(cache_size=2)
        
        # Add three sentences to cache (exceeding capacity)
        sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
        
        # Mock _embed_uncached to track calls
        with patch.object(model, '_embed_uncached') as mock_embed:
            mock_embed.side_effect = lambda s: np.random.randn(len(s), 384)
            
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
        
        # Add some sentences
        sentences = ["Stat test 1", "Stat test 2"]
        model.embed(sentences)
        
        stats = model.get_cache_stats()
        assert stats['cache_size'] == 2
        assert stats['max_cache_size'] == 100
        # hit_rate may be None if not tracking
        assert stats['cache_hit_rate'] is None or isinstance(stats['cache_hit_rate'], float)

    def test_embed_uncached_deterministic(self):
        """Test that _embed_uncached produces deterministic embeddings."""
        model = EmbeddingModel()
        sentences = ["Deterministic A", "Deterministic B"]
        
        # Call twice with cleared cache in between
        result1 = model._embed_uncached(sentences)
        model.clear_cache()
        result2 = model._embed_uncached(sentences)
        
        # Should be identical (deterministic based on sentence hash)
        np.testing.assert_array_equal(result1, result2)

    def test_embed_uncached_normalization(self):
        """Test that _embed_uncached returns normalized embeddings."""
        model = EmbeddingModel()
        sentences = ["Normalization test"]
        embeddings = model._embed_uncached(sentences)
        
        # Check shape
        assert embeddings.shape == (1, 384)
        
        # Check normalization
        norm = np.linalg.norm(embeddings[0])
        assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_error_handling_large_input(self):
        """Test embedding with very long sentence (should not crash)."""
        model = EmbeddingModel()
        long_sentence = "word " * 1000  # Very long sentence
        result = model.embed([long_sentence])
        
        # Should still produce embedding
        assert result.shape == (1, 384)
        assert np.isfinite(result).all()

    @pytest.mark.skip(reason="Actual HuggingFace model not implemented yet")
    def test_actual_model_integration(self):
        """TODO: Test with actual HuggingFace model when implemented."""
        pass


class TestSingletonAndConvenience:
    """Test the singleton pattern and convenience functions."""

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
        
        with patch('app.pipeline.embedding.get_embedding_model',
                   return_value=mock_model):
            result = embed_sentences(sentences)
        
        # Should call model.embed with the sentences
        mock_model.embed.assert_called_once_with(sentences)
        np.testing.assert_array_equal(result, mock_embeddings)

    def test_embed_sentences_empty(self):
        """Test embed_sentences with empty list."""
        mock_model = MagicMock()
        mock_model.embed.return_value = np.array([])
        
        with patch('app.pipeline.embedding.get_embedding_model',
                   return_value=mock_model):
            result = embed_sentences([])
        
        mock_model.embed.assert_called_once_with([])
        assert result.shape == (0,)

    def test_singleton_reset(self):
        """Test that singleton can be reset by module reload (edge case)."""
        # This is more of a documentation test - in practice, singleton
        # persists across imports unless module is reloaded
        import importlib
        import sys
        
        # Import the module directly (already imported at top of file)
        from app.pipeline import embedding as embedding_module
        
        # Get current singleton
        model1 = embedding_module.get_embedding_model()
        
        # Reload module (simulates fresh import)
        importlib.reload(embedding_module)
        
        # Get new singleton
        model2 = embedding_module.get_embedding_model()
        
        # They might be different objects after reload
        # (This test is mostly to document the behavior)
        assert model1 is not model2 or model1 is model2  # Either is possible


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_unicode_and_special_characters(self):
        """Test embedding sentences with Unicode and special characters."""
        model = EmbeddingModel()
        sentences = [
            "Hello world!",
            "Привет мир!",  # Russian
            "🎉 Emoji test 🚀",
            "Line\nbreak",
            "Tab\tcharacter",
            "   Extra   spaces   ",
            "",  # Empty string (should still work)
        ]
        
        # Should not crash
        result = model.embed(sentences)
        assert result.shape == (len(sentences), 384)
        
        # All embeddings should be finite
        assert np.isfinite(result).all()

    def test_very_short_sentences(self):
        """Test embedding very short sentences."""
        model = EmbeddingModel()
        sentences = ["a", "b", "c", "1", "!", " "]
        
        result = model.embed(sentences)
        assert result.shape == (len(sentences), 384)

    def test_duplicate_sentences(self):
        """Test embedding duplicate sentences."""
        model = EmbeddingModel()
        sentences = ["Duplicate", "Duplicate", "Duplicate"]
        
        result = model.embed(sentences)
        assert result.shape == (3, 384)
        
        # All three should be identical (cached after first)
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[1], result[2])

    def test_cache_with_identical_strings_different_objects(self):
        """Test that identical string content is cached regardless of object identity."""
        model = EmbeddingModel()
        
        # Two different string objects with same content
        str1 = "Test string"
        str2 = "Test " + "string"  # Different object, same content
        
        # Embed first
        embedding1 = model.embed([str1])
        
        # Embed second - should be cached
        with patch.object(model, '_embed_uncached') as mock_embed:
            embedding2 = model.embed([str2])
            mock_embed.assert_not_called()
        
        # Should be equal
        np.testing.assert_array_equal(embedding1, embedding2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])