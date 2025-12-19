"""
Tests for sophisticated insight generation (domain patterns, nuanced sentiment, etc.).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.app.pipeline.insights import InsightGenerator


class TestSophisticatedInsights:
    """Test sophisticated insight generation features."""
    
    def setup_method(self):
        self.generator = InsightGenerator()
        self.mock_preprocessor = MagicMock()
        # Default mock behavior
        self.mock_preprocessor.extract_key_terms.return_value = []
        
        # Helper for creating dummy embeddings
        self.dummy_embedding = np.random.rand(1, 384)

    def test_domain_pattern_detection(self):
        """Test detection of domain patterns from keywords."""
        sentences = [
            "The price is too high for this service.",
            "It costs a lot of money.",
            "Expensive subscription fees."
        ]
        embeddings = np.random.rand(len(sentences), 384)
        sentiment_info = {'sentiment': 'negative', 'confidence': 0.8}
        
        # Mock key terms that reinforce the domain
        self.mock_preprocessor.extract_key_terms.return_value = ['price', 'expensive']
        
        result = self.generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=self.mock_preprocessor
        )
        
        assert result is not None
        assert result['domain'] == 'Pricing'
        # Check if domain insight is generated
        assert any("Pricing" in insight for insight in result['insights']) or \
               any("price" in insight.lower() for insight in result['insights'])

    def test_nuanced_sentiment_mixed(self):
        """Test generation of mixed sentiment insight."""
        sentences = ["Mixed feelings."] * 10
        embeddings = np.random.rand(10, 384)
        
        # Mixed sentiment profile
        sentiment_info = {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'proportions': {'positive': 0.4, 'negative': 0.4, 'neutral': 0.2},
            'avg_compound': 0.05
        }
        
        result = self.generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=self.mock_preprocessor
        )
        
        assert result is not None
        # Should detect mixed sentiment
        insights = result['insights']
        assert any("Mixed sentiment" in i for i in insights) or \
               any("Polarized" in i for i in insights)

    def test_cluster_size_insights_large(self):
        """Test insight generation for large clusters relative to dataset."""
        sentences = ["Sentence"] * 25
        embeddings = np.random.rand(25, 384)
        sentiment_info = {'sentiment': 'positive', 'confidence': 0.9}
        total_dataset_size = 100 # 25% of dataset
        
        result = self.generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=self.mock_preprocessor,
            total_dataset_size=total_dataset_size
        )
        
        assert result is not None
        insights = result['insights']
        # Should mention percentage or "major theme"
        assert any("25.0%" in i for i in insights) or \
               any("major theme" in i.lower() for i in insights)

    def test_cluster_size_insights_niche(self):
        """Test insight generation for niche/small clusters."""
        sentences = ["Sentence"] * 2
        embeddings = np.random.rand(2, 384)
        sentiment_info = {'sentiment': 'neutral'}
        total_dataset_size = 100 # 2% of dataset
        
        result = self.generator.generate_cluster_insights(
            cluster_id=1,
            sentences=sentences,
            embeddings=embeddings,
            sentiment_info=sentiment_info,
            preprocessor=self.mock_preprocessor,
            total_dataset_size=total_dataset_size
        )
        
        assert result is not None
        insights = result['insights']
        # Should mention "niche" or "small"
        assert any("niche" in i.lower() for i in insights) or \
               any("isolated" in i.lower() for i in insights)

    def test_domain_identification_logic(self):
        """Directly test the domain identification helper."""
        # Test Pricing
        assert self.generator._identify_domain(
            ["The cost is high"], ['expensive']
        ) == 'Pricing'
        
        # Test Quality
        assert self.generator._identify_domain(
            ["It broke immediately"], ['quality', 'defect']
        ) == 'Quality'
        
        # Test UX/UI
        assert self.generator._identify_domain(
            ["Hard to navigate the app"], ['interface', 'screen']
        ) == 'UX/UI'
        
        # Test None
        assert self.generator._identify_domain(
            ["Random sentence about nothing"], ['cats', 'dogs']
        ) is None
