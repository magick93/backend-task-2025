# Text Analysis Microservice - Python Implementation Details

## 1. Python Code Structure and Modular Architecture

### 1.1 Package Structure

```
lambda_function/
├── __init__.py
├── handler.py                  # Main Lambda handler
├── config.py                   # Configuration management
├── models/                     # Data models and schemas
│   ├── __init__.py
│   ├── schemas.py             # Pydantic schemas for input/output
│   ├── data_models.py         # Internal data structures
│   └── enums.py               # Enum definitions
├── pipeline/                   # Core processing pipeline
│   ├── __init__.py
│   ├── preprocessor.py        # Text preprocessing
│   ├── embedder.py            # Sentence embeddings
│   ├── clusterer.py           # Clustering algorithms
│   ├── sentiment.py           # Sentiment analysis
│   ├── insights.py            # Insight generation
│   ├── comparator.py          # Comparative analysis
│   └── pipeline.py            # Main pipeline orchestration
├── cache/                      # Caching layer
│   ├── __init__.py
│   ├── redis_client.py        # Redis caching
│   └── cache_manager.py       # Cache management
├── storage/                    # AWS storage integration
│   ├── __init__.py
│   ├── s3_client.py           # S3 operations
│   ├── dynamodb_client.py     # DynamoDB operations
│   └── storage_manager.py     # Storage orchestration
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── validation.py          # Input validation
│   ├── logging.py             # Structured logging
│   ├── metrics.py             # CloudWatch metrics
│   ├── error_handling.py      # Error handling utilities
│   └── performance.py         # Performance monitoring
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── performance/
└── requirements.txt           # Dependencies
```

### 1.2 Module Design Principles

- **Single Responsibility**: Each module handles one specific concern
- **Dependency Injection**: External dependencies injected for testability
- **Configuration-driven**: All parameters configurable via environment variables
- **Type Safety**: Comprehensive type hints using Python 3.12 typing
- **Async Support**: Async operations for I/O-bound tasks

## 2. Core Implementation Components

### 2.1 Input Validation and Schema Definitions

```python
# models/schemas.py
from pydantic import BaseModel, Field, validator, constr
from typing import List, Optional, Literal
from enum import Enum

class Sentence(BaseModel):
    """Individual sentence with unique identifier."""
    sentence: str = Field(..., min_length=1, max_length=1000)
    id: constr(regex=r'^[a-zA-Z0-9\-_]+$')  # Alphanumeric with hyphens/underscores
    
    @validator('sentence')
    def validate_sentence_length(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Sentence cannot be empty or whitespace only')
        return v.strip()

class AnalysisType(str, Enum):
    STANDALONE = "standalone"
    COMPARATIVE = "comparative"

class StandaloneAnalysisRequest(BaseModel):
    """Request schema for standalone analysis."""
    surveyTitle: str = Field(..., min_length=1, max_length=200)
    theme: str = Field(..., min_length=1, max_length=100)
    baseline: List[Sentence] = Field(..., min_items=1, max_items=1000)
    
    @validator('baseline')
    def validate_unique_ids(cls, v):
        ids = [sentence.id for sentence in v]
        if len(ids) != len(set(ids)):
            raise ValueError('Sentence IDs must be unique')
        return v

class ComparativeAnalysisRequest(BaseModel):
    """Request schema for comparative analysis."""
    surveyTitle: str = Field(..., min_length=1, max_length=200)
    theme: str = Field(..., min_length=1, max_length=100)
    baseline: List[Sentence] = Field(..., min_items=1, max_items=1000)
    comparison: List[Sentence] = Field(..., min_items=1, max_items=1000)

class AnalysisResponse(BaseModel):
    """Base response schema."""
    jobId: str
    status: Literal["processing", "completed", "failed"]
    processingTimeMs: Optional[int] = None
    resultUrl: Optional[str] = None
    error: Optional[str] = None

class ClusterInsight(BaseModel):
    """Individual cluster insight."""
    title: str
    representativeSentences: List[str]
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    keyTerms: List[str]

class StandaloneAnalysisResult(BaseModel):
    """Result schema for standalone analysis."""
    clusters: List[ClusterInsight]
    summary: dict
    metadata: dict

class ComparativeAnalysisResult(BaseModel):
    """Result schema for comparative analysis."""
    baselineClusters: List[ClusterInsight]
    comparisonClusters: List[ClusterInsight]
    keySimilarities: List[dict]
    keyDifferences: List[dict]
    similarityScore: float = Field(..., ge=0.0, le=1.0)
```

### 2.2 Text Preprocessing Pipeline

```python
# pipeline/preprocessor.py
import re
import string
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """Text preprocessing with configurable steps."""
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        
        # Download NLTK resources if not present
        self._ensure_nltk_resources()
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are available."""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    
    def preprocess(self, text: str) -> str:
        """Preprocess a single text string."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            # Keep apostrophes for contractions
            text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Rejoin
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]
    
    def extract_key_terms(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key terms using TF-IDF or RAKE."""
        # Simplified implementation - in production would use sklearn or RAKE
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Count frequencies
        from collections import Counter
        term_freq = Counter(tokens)
        
        # Return top N terms
        return [term for term, _ in term_freq.most_common(top_n)]
```

### 2.3 Sentence Embedding Generation

```python
# pipeline/embedder.py
import numpy as np
from typing import List, Optional, Tuple
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
from cache.redis_client import RedisCache

class EmbeddingService:
    """Service for generating and caching sentence embeddings."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_enabled: bool = True,
                 cache_ttl: int = 86400):  # 24 hours
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        
        # Lazy loading of model
        self._model = None
        self._cache = RedisCache() if cache_enabled else None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model to reduce cold start time."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for a sentence."""
        # Use hash of normalized text
        normalized = text.strip().lower()
        return f"embedding:{hashlib.sha256(normalized.encode()).hexdigest()}"
    
    def get_embeddings(self, 
                       sentences: List[str], 
                       batch_size: int = 32) -> np.ndarray:
        """
        Get embeddings for sentences, using cache when available.
        
        Args:
            sentences: List of sentence strings
            batch_size: Batch size for model inference
            
        Returns:
            numpy array of embeddings with shape (n_sentences, embedding_dim)
        """
        if not sentences:
            return np.array([])
        
        embeddings = np.zeros((len(sentences), self.model.get_sentence_embedding_dimension()))
        uncached_indices = []
        uncached_sentences = []
        
        # Check cache for each sentence
        for i, sentence in enumerate(sentences):
            if self.cache_enabled:
                cached = self._cache.get(self._generate_cache_key(sentence))
                if cached is not None:
                    embeddings[i] = pickle.loads(cached)
                else:
                    uncached_indices.append(i)
                    uncached_sentences.append(sentence)
            else:
                uncached_indices.append(i)
                uncached_sentences.append(sentence)
        
        # Generate embeddings for uncached sentences
        if uncached_sentences:
            new_embeddings = self.model.encode(
                uncached_sentences,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Store in cache and update result array
            for idx, sentence, embedding in zip(uncached_indices, 
                                                uncached_sentences, 
                                                new_embeddings):
                embeddings[idx] = embedding
                
                if self.cache_enabled:
                    self._cache.set(
                        self._generate_cache_key(sentence),
                        pickle.dumps(embedding),
                        ex=self.cache_ttl
                    )
        
        return embeddings
    
    def get_similarity_matrix(self, 
                              embeddings1: np.ndarray, 
                              embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of embeddings."""
        # Normalize embeddings
        norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms1 = np.where(norms1 == 0, 1, norms1)
        norms2 = np.where(norms2 == 0, 1, norms2)
        
        embeddings1_norm = embeddings1 / norms1
        embeddings2_norm = embeddings2 / norms2
        
        # Compute similarity matrix
        similarity = np.dot(embeddings1_norm, embeddings2_norm.T)
        
        # Clip to valid range due to floating point errors
        return np.clip(similarity, -1.0, 1.0)
```

### 2.4 Clustering Implementation (HDBSCAN)

```python
# pipeline/clusterer.py
import numpy as np
from typing import Tuple, List, Optional, Dict
import hdbscan
import umap
from sklearn.metrics import silhouette_score
import warnings

class ClusteringService:
    """Clustering service using HDBSCAN with UMAP dimensionality reduction."""
    
    def __init__(self,
                 min_cluster_size: int = 3,
                 min_samples: int = 2,
                 cluster_selection_epsilon: float = 0.0,
                 metric: str = 'euclidean',
                 n_components: int = 50,
                 random_state: int = 42):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.n_components = n_components
        self.random_state = random_state
        
        # Initialize UMAP reducer
        self.reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
    
    def cluster(self, 
                embeddings: np.ndarray,
                reduce_dimensionality: bool = True) -> Tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
        """
        Cluster embeddings using HDBSCAN.
        
        Args:
            embeddings: Array of embeddings with shape (n_samples, embedding_dim)
            reduce_dimensionality: Whether to apply UMAP dimensionality reduction
            
        Returns:
            Tuple of (labels, probabilities, clusterer)
            - labels: Cluster labels (-1 for noise)
            - probabilities: Cluster membership probabilities
            - clusterer: Fitted HDBSCAN object
        """
        if len(embeddings) < self.min_cluster_size:
            # Not enough samples for clustering
            return np.full(len(embeddings), -1), np.zeros(len(embeddings)), None
        
        # Reduce dimensionality for better clustering performance
        if reduce_dimensionality and embeddings.shape[1] > self.n_components:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reduced_embeddings = self.reducer.fit_transform(embeddings)
        else:
            reduced_embeddings = embeddings
        
        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            prediction_data=True  # Enable prediction for new points
        )
        
        labels = clusterer.fit_predict(reduced_embeddings)
        probabilities = clusterer.probabilities_
        
        return labels, probabilities, clusterer
    
    def evaluate_clustering(self, 
                           embeddings: np.ndarray, 
                           labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise_points': np.sum(labels == -1),
            'noise_ratio': np.sum(labels == -1) / len(labels) if len(labels) > 0 else 0
        }
        
        # Calculate silhouette score (excluding noise points)
        valid_indices = labels != -1
        if np.sum(valid_indices) > 1 and n_clusters > 1:
            try:
                silhouette = silhouette_score(
                    embeddings[valid_indices],
                    labels[valid_indices]
                )
                metrics['silhouette_score'] = silhouette
            except:
                metrics['silhouette_score'] = -1.0
        else:
            metrics['silhouette_score'] = -1.0
        
        return metrics
    
    def get_cluster_centroids(self, 
                             embeddings: np.ndarray, 
                             labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculate centroids for each cluster."""
        centroids = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 0:
                centroids[label] = np.mean(cluster_points, axis=0)
        
        return centroids
    
    def get_representative_sentences(self,
                                    sentences: List[str],
                                    embeddings: np.ndarray,
                                    labels: np.ndarray,
                                    centroids: Dict[int, np.ndarray],
                                    n_per_cluster: int = 3) -> Dict[int, List[str]]:
        """Get representative sentences closest to cluster centroids."""
        representative = {}
        
        for label, centroid in centroids.items():
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) == 0:
                continue
            
            # Calculate distances to centroid
            cluster_embeddings = embeddings[cluster_indices]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            
            # Get indices of closest sentences
            closest_indices = np.argsort(distances)[:n_per_cluster]
            representative[label] = [
                sentences[cluster_indices[idx]] for idx in closest_indices
            ]
        
        return representative

### 2.5 Sentiment Analysis

```python
# pipeline/sentiment.py
from typing import List, Dict, Tuple, Optional
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

class SentimentAnalyzer:
    """Sentiment analysis using DistilBERT (primary) and VADER (fallback)."""
    
    def __init__(self,
                 model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 use_vader_fallback: bool = True,
                 confidence_threshold: float = 0.7):
        self.model_name = model_name
        self.use_vader_fallback = use_vader_fallback
        self.confidence_threshold = confidence_threshold
        
        # Lazy loading of models
        self._transformer_pipeline = None
        self._vader_analyzer = None
        self._tokenizer = None
        self._model = None
    
    @property
    def transformer_pipeline(self):
        """Lazy load transformer pipeline."""
        if self._transformer_pipeline is None:
            self._transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512
            )
        return self._transformer_pipeline
    
    @property
    def vader_analyzer(self):
        """Lazy load VADER analyzer."""
        if self._vader_analyzer is None and self.use_vader_fallback:
            self._vader_analyzer = SentimentIntensityAnalyzer()
        return self._vader_analyzer
    
    def analyze_sentence(self, sentence: str) -> Dict[str, any]:
        """
        Analyze sentiment of a single sentence.
        
        Returns:
            Dictionary with keys: label, confidence, scores, method
        """
        # Try transformer model first
        try:
            result = self.transformer_pipeline(sentence)[0]
            label = result['label'].lower()
            confidence = result['score']
            
            # Map transformer labels to our standard labels
            if label in ['positive', 'pos']:
                mapped_label = 'positive'
            elif label in ['negative', 'neg']:
                mapped_label = 'negative'
            else:
                mapped_label = 'neutral'
            
            # Use VADER as fallback if confidence is low
            if confidence < self.confidence_threshold and self.use_vader_fallback:
                vader_result = self._analyze_with_vader(sentence)
                if vader_result['confidence'] > confidence:
                    return vader_result
            
            return {
                'label': mapped_label,
                'confidence': confidence,
                'scores': {'transformer': confidence},
                'method': 'transformer'
            }
            
        except Exception as e:
            # Fallback to VADER if transformer fails
            if self.use_vader_fallback:
                warnings.warn(f"Transformer sentiment analysis failed: {e}. Using VADER fallback.")
                return self._analyze_with_vader(sentence)
            else:
                raise
    
    def _analyze_with_vader(self, sentence: str) -> Dict[str, any]:
        """Analyze sentiment using VADER."""
        scores = self.vader_analyzer.polarity_scores(sentence)
        
        # Determine label based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
            confidence = compound
        elif compound <= -0.05:
            label = 'negative'
            confidence = abs(compound)
        else:
            label = 'neutral'
            confidence = 1.0 - abs(compound)
        
        return {
            'label': label,
            'confidence': confidence,
            'scores': scores,
            'method': 'vader'
        }
    
    def analyze_batch(self, sentences: List[str]) -> List[Dict[str, any]]:
        """Analyze sentiment for a batch of sentences."""
        return [self.analyze_sentence(sentence) for sentence in sentences]
    
    def get_cluster_sentiment(self, 
                             sentences: List[str],
                             embeddings: np.ndarray,
                             labels: np.ndarray) -> Dict[int, Dict[str, any]]:
        """
        Calculate aggregate sentiment for each cluster.
        
        Args:
            sentences: List of all sentences
            embeddings: Sentence embeddings
            labels: Cluster labels (-1 for noise)
            
        Returns:
            Dictionary mapping cluster label to sentiment analysis
        """
        cluster_sentiments = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            
            cluster_indices = np.where(labels == label)[0]
            cluster_sentences = [sentences[i] for i in cluster_indices]
            
            if not cluster_sentences:
                continue
            
            # Analyze each sentence in cluster
            sentiment_results = self.analyze_batch(cluster_sentences)
            
            # Aggregate sentiment
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0}
            confidences = []
            
            for result in sentiment_results:
                sentiment_counts[result['label']] += 1
                confidences.append(result['confidence'])
            
            # Determine dominant sentiment
            total = len(sentiment_results)
            if total == 0:
                continue
            
            # Calculate proportions
            proportions = {k: v/total for k, v in sentiment_counts.items()}
            
            # Determine overall sentiment
            max_prop = max(proportions.values())
            dominant_sentiments = [k for k, v in proportions.items() if v == max_prop]
            
            if len(dominant_sentiments) == 1:
                overall_sentiment = dominant_sentiments[0]
            else:
                overall_sentiment = 'mixed'
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            cluster_sentiments[label] = {
                'sentiment': overall_sentiment,
                'confidence': avg_confidence,
                'proportions': proportions,
                'counts': sentiment_counts,
                'total_sentences': total
            }
        
        return cluster_sentiments
```

### 2.6 Insight Generation Logic

```python
# pipeline/insights.py
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter
import re

class InsightGenerator:
    """Generate insights from clustered sentences."""
    
    def __init__(self, 
                 max_insights_per_cluster: int = 5,
                 min_sentence_length: int = 10,
                 max_title_length: int = 100):
        self.max_insights_per_cluster = max_insights_per_cluster
        self.min_sentence_length = min_sentence_length
        self.max_title_length = max_title_length
    
    def generate_cluster_insights(self,
                                 cluster_id: int,
                                 sentences: List[str],
                                 embeddings: np.ndarray,
                                 sentiment_info: Dict[str, Any],
                                 preprocessor) -> Dict[str, Any]:
        """
        Generate insights for a single cluster.
        
        Args:
            cluster_id: Cluster identifier
            sentences: Sentences in the cluster
            embeddings: Sentence embeddings
            sentiment_info: Sentiment analysis results for the cluster
            preprocessor: TextPreprocessor instance
            
        Returns:
            Dictionary with cluster insights
        """
        if not sentences:
            return None
        
        # Extract key terms
        all_text = ' '.join(sentences)
        key_terms = preprocessor.extract_key_terms(all_text, top_n=10)
        
        # Generate cluster title
        title = self._generate_cluster_title(sentences, key_terms)
        
        # Get representative sentences (closest to centroid)
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_indices = np.argsort(distances)[:3]
        representative_sentences = [sentences[i] for i in closest_indices]
        
        # Calculate sentence diversity
        diversity_score = self._calculate_diversity(embeddings)
        
        # Identify common patterns
        patterns = self._identify_patterns(sentences)
        
        return {
            'cluster_id': cluster_id,
            'title': title,
            'representative_sentences': representative_sentences,
            'sentiment': sentiment_info.get('sentiment', 'neutral'),
            'sentiment_confidence': sentiment_info.get('confidence', 0.0),
            'key_terms': key_terms,
            'sentence_count': len(sentences),
            'diversity_score': diversity_score,
            'patterns': patterns,
            'summary': self._generate_summary(sentences, sentiment_info)
        }
    
    def _generate_cluster_title(self, 
                               sentences: List[str], 
                               key_terms: List[str]) -> str:
        """Generate a descriptive title for the cluster."""
        # Use the most frequent key terms
        if key_terms:
            # Look for sentences containing key terms
            for sentence in sentences:
                for term in key_terms[:3]:
                    if term.lower() in sentence.lower():
                        # Extract a concise phrase around the term
                        words = sentence.split()
                        if len(words) <= 8:
                            return sentence[:self.max_title_length]
                        
                        # Find term position and extract surrounding words
                        try:
                            idx = words.index(term)
                            start = max(0, idx - 2)
                            end = min(len(words), idx + 3)
                            title = ' '.join(words[start:end])
                            return title[:self.max_title_length]
                        except ValueError:
                            continue
        
        # Fallback: use first sentence truncated
        first_sentence = sentences[0]
        if len(first_sentence) > self.max_title_length:
            return first_sentence[:self.max_title_length-3] + '...'
        return first_sentence
    
    def _calculate_diversity(self, embeddings: np.ndarray) -> float:
        """Calculate diversity score based on embedding variance."""
        if len(embeddings) <= 1:
            return 0.0
        
        # Calculate pairwise distances
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(embeddings)
        
        # Diversity is average pairwise distance
        diversity = np.mean(distances)
        
        # Normalize to 0-1 range
        return float(np.clip(diversity, 0.0, 1.0))
    
    def _identify_patterns(self, sentences: List[str]) -> List[str]:
        """Identify common linguistic patterns in sentences."""
        patterns = []
        
        # Look for common sentence starters
        starters = Counter()
        for sentence in sentences:
            first_word = sentence.split()[0].lower() if sentence.split() else ''
            if first_word:
                starters[first_word] += 1
        
        # Add patterns for frequent starters
        for word, count in starters.most_common(3):
            if count >= 2:
                patterns.append(f"Often starts with '{word}'")
        
        # Look for common phrases (bigrams)
        all_words = ' '.join(sentences).lower().split()
        bigrams = Counter()
        for i in range(len(all_words) - 1):
            bigram = f"{all_words[i]} {all_words[i+1]}"
            bigrams[bigram] += 1
        
        # Add frequent bigrams
        for bigram, count in bigrams.most_common(3):
            if count >= 2:
                patterns.append(f"Frequent phrase: '{bigram}'")
        
        return patterns
    
    def _generate_summary(self, 
                         sentences: List[str], 
                         sentiment_info: Dict[str, Any]) -> str:
        """Generate a natural language summary of the cluster."""
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
    
    def generate_overall_summary(self,
                                clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall summary across all clusters."""
        if not clusters:
            return {'total_clusters': 0, 'total_sentences': 0}
        
        total_sentences = sum(c['sentence_count'] for c in clusters)
        sentiment_distribution = Counter()
        
        for cluster in clusters:
            sentiment_distribution[cluster['sentiment']] += cluster['sentence_count']
        
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
        largest_cluster = max(clusters, key=lambda x: x['sentence_count'])
        
        return {
            'total_clusters': len(clusters),
            'total_sentences': total_sentences,
            'sentiment_distribution': dict(sentiment_distribution),
            'sentiment_percentages': sentiment_percentages,
            'dominant_sentiment': dominant,
            'largest_cluster': {
                'title': largest_cluster['title'],
                'size': largest_cluster['sentence_count'],
                'sentiment': largest_cluster['sentiment']
            },
            'average_cluster_size': total_sentences / len(clusters) if clusters else 0
        }
```

### 2.7 Output Formatting

```python
# pipeline/pipeline.py
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime
from models.schemas import (
    StandaloneAnalysisResult, 
    ComparativeAnalysisResult,
    ClusterInsight
)

class OutputFormatter:
    """Format analysis results into standardized output."""
    
    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata
    
    def format_standalone_result(self,
                                clusters: List[Dict[str, Any]],
                                overall_summary: Dict[str, Any],
                                processing_time_ms: int,
                                job_id: str) -> StandaloneAnalysisResult:
        """Format standalone analysis results."""
        # Convert clusters to ClusterInsight objects
        cluster_insights = []
        for cluster in clusters:
            insight = ClusterInsight(
                title=cluster['title'],
                representativeSentences=cluster['representative_sentences'],
                sentiment=cluster['sentiment'],
                confidence=cluster['sentiment_confidence'],
                keyTerms=cluster['key_terms']
            )
            cluster_insights.append(insight)
        
        # Prepare metadata
        metadata = {
            'jobId': job_id,
            'processingTimeMs': processing_time_ms,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'clusterCount': len(clusters),
            'totalSentences': overall_summary['total_sentences']
        }
        
        return StandaloneAnalysisResult(
            clusters=cluster_insights,
            summary=overall_summary,
            metadata=metadata
        )
    
    def format_comparative_result(self,
                                 baseline_clusters: List[Dict[str, Any]],
                                 comparison_clusters: List[Dict[str, Any]],
                                 similarities: List[Dict[str, Any]],
                                 differences: List[Dict[str, Any]],
                                 similarity_score: float,
                                 processing_time_ms: int,
                                 job_id: str) -> ComparativeAnalysisResult:
        """Format comparative analysis results."""
        # Convert baseline clusters
        baseline_insights = []
        for cluster in baseline_clusters:
            insight = ClusterInsight(
                title=cluster['title'],
                representativeSentences=cluster['representative_sentences'],
                sentiment=cluster['sentiment'],
                confidence=cluster['sentiment_confidence'],
                keyTerms=cluster['key_terms']
            )
            baseline_insights.append(insight)
        
        # Convert comparison clusters
        comparison_insights = []
        for cluster in comparison_clusters:
            insight = ClusterInsight(
                title=cluster['title'],
                representativeSentences=cluster['representative_sentences'],
                sentiment=cluster['sentiment'],
                confidence=cluster['sentiment_confidence'],
                keyTerms=cluster['key_terms']
            )
            comparison_insights.append(insight)
        
        return ComparativeAnalysisResult(
            baselineClusters=baseline_insights,
            comparisonClusters=comparison_insights,
            keySimilarities=similarities,
            keyDifferences=differences,
            similarityScore=similarity_score
        )
    
    def format_error_response(self,
                             error: Exception,
                             job_id: str,
                             processing_time_ms: int) -> Dict[str, Any]:
        """Format error response."""
        error_type = type(error).__name__
        error_message = str(error)
        
        return {
            'jobId': job_id,
            'status': 'failed',
            'processingTimeMs': processing_time_ms,
            'error': f'{error_type}: {error_message}',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    def to_json(self, result: Any) -> str:
        """Convert result to JSON string."""
        if hasattr(result, 'dict'):
            return json.dumps(result.dict(), indent=2, ensure_ascii=False)
        else:
            return json.dumps(result, indent=2, ensure_ascii=False)
```

### 2.8 Main Pipeline Orchestration

```python
# pipeline/pipeline.py
from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime
from