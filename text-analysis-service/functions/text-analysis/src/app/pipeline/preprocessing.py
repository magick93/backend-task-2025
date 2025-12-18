"""
Text preprocessing utilities.

Provides text cleaning, normalization, and utility functions for
preparing text data for embedding and analysis.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import re
import string
from collections import Counter
import unicodedata

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class TextPreprocessor:
    """
    Text preprocessing utilities for cleaning and normalizing text.
    
    Provides:
    - Text cleaning (remove special characters, normalize whitespace)
    - Stop word removal (optional)
    - Text normalization (lowercasing, unicode normalization)
    - Sentence splitting
    - Basic text statistics
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        min_sentence_length: int = 3,
        max_sentence_length: int = 1000
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation characters
            remove_numbers: Remove numeric digits
            remove_extra_whitespace: Normalize whitespace
            normalize_unicode: Normalize unicode characters
            min_sentence_length: Minimum sentence length to keep
            max_sentence_length: Maximum sentence length to keep
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        # Compile regex patterns for efficiency
        self._whitespace_pattern = re.compile(r'\s+')
        self._punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        self._number_pattern = re.compile(r'\d+')
        
        logger.debug("TextPreprocessor initialized")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        original_length = len(text)
        processed = text
        
        # 1. Normalize unicode
        if self.normalize_unicode:
            processed = unicodedata.normalize('NFKD', processed)
            processed = processed.encode('ascii', 'ignore').decode('ascii')
        
        # 2. Convert to lowercase
        if self.lowercase:
            processed = processed.lower()
        
        # 3. Remove punctuation
        if self.remove_punctuation:
            processed = self._punctuation_pattern.sub(' ', processed)
        
        # 4. Remove numbers
        if self.remove_numbers:
            processed = self._number_pattern.sub(' ', processed)
        
        # 5. Remove extra whitespace
        if self.remove_extra_whitespace:
            processed = self._whitespace_pattern.sub(' ', processed)
            processed = processed.strip()
        
        # Log if significant reduction occurred
        if original_length > 0 and len(processed) < original_length * 0.5:
            logger.debug(
                f"Text reduced from {original_length} to {len(processed)} "
                f"characters ({len(processed)/original_length:.0%})"
            )
        
        return processed
    
    def preprocess_sentences(self, sentences: List[str]) -> List[str]:
        """
        Preprocess a list of sentences.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of preprocessed sentences
        """
        if not sentences:
            return []
        
        logger.debug(f"Preprocessing {len(sentences)} sentences")
        
        processed = []
        for i, sentence in enumerate(sentences):
            try:
                cleaned = self.preprocess_text(sentence)
                
                # Filter by length
                if (len(cleaned) >= self.min_sentence_length and 
                    len(cleaned) <= self.max_sentence_length):
                    processed.append(cleaned)
                else:
                    logger.debug(
                        f"Sentence {i} filtered out: length {len(cleaned)} "
                        f"(min={self.min_sentence_length}, max={self.max_sentence_length})"
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to preprocess sentence {i}: {str(e)}")
                # Keep original as fallback
                processed.append(sentence)
        
        logger.debug(f"Preprocessing complete: {len(sentences)} -> {len(processed)} sentences")
        return processed
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple heuristic.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitting by punctuation
        # TODO: Replace with more sophisticated sentence tokenizer (nltk, spaCy)
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        cleaned = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned.append(sentence)
        
        return cleaned
    
    def remove_stop_words(self, text: str) -> str:
        """
        Remove stop words from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stop words removed
        """
        if not text:
            return ""
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        
        return ' '.join(filtered_words)
    
    def extract_key_terms(
        self, 
        text: str, 
        top_n: int = 10,
        min_word_length: int = 3,
        exclude_stop_words: bool = True
    ) -> List[str]:
        """
        Extract key terms from text based on frequency.
        
        Args:
            text: Input text
            top_n: Number of top terms to return
            min_word_length: Minimum word length to consider
            exclude_stop_words: Whether to exclude stop words
            
        Returns:
            List of key terms
        """
        if not text:
            return []
        
        # Preprocess text
        processed = self.preprocess_text(text)
        
        # Split into words
        words = processed.split()
        
        # Filter words
        filtered_words = []
        for word in words:
            if len(word) < min_word_length:
                continue
            if exclude_stop_words and word.lower() in self.stop_words:
                continue
            filtered_words.append(word)
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get top terms
        top_terms = [
            word for word, _ in 
            word_counts.most_common(top_n)
        ]
        
        return top_terms
    
    def calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Calculate basic text statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0
            }
        
        # Basic counts
        char_count = len(text)
        word_count = len(text.split())
        sentences = self.split_into_sentences(text)
        sentence_count = len(sentences)
        
        # Average word length
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
        else:
            avg_word_length = 0.0
        
        # Average sentence length (in words)
        if sentences:
            sentence_word_counts = [len(sentence.split()) for sentence in sentences]
            avg_sentence_length = sum(sentence_word_counts) / len(sentence_word_counts)
        else:
            avg_sentence_length = 0.0
        
        # Vocabulary richness (unique words / total words)
        if word_count > 0:
            unique_words = len(set(words))
            vocabulary_richness = unique_words / word_count
        else:
            vocabulary_richness = 0.0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'vocabulary_richness': round(vocabulary_richness, 3),
            'unique_word_count': len(set(words)) if words else 0
        }
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts (alias for preprocess_sentences).
        
        Args:
            texts: List of text strings (typically sentences)
            
        Returns:
            List of preprocessed texts
        """
        return self.preprocess_sentences(texts)
    
    def batch_preprocess(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[str]:
        """
        Preprocess a batch of texts with optional progress tracking.
        
        Args:
            texts: List of text strings
            show_progress: Whether to log progress
            
        Returns:
            List of preprocessed texts
        """
        if not texts:
            return []
        
        logger.info(f"Batch preprocessing {len(texts)} texts")
        
        processed = []
        for i, text in enumerate(texts):
            if show_progress and i % 100 == 0:
                logger.debug(f"Preprocessing progress: {i}/{len(texts)}")
            
            try:
                cleaned = self.preprocess_text(text)
                processed.append(cleaned)
            except Exception as e:
                logger.warning(f"Failed to preprocess text {i}: {str(e)}")
                processed.append("")  # Empty string as fallback
        
        # Filter out empty strings
        filtered = [text for text in processed if text]
        
        logger.info(
            f"Batch preprocessing complete: {len(texts)} -> {len(filtered)} "
            f"non-empty texts ({len(filtered)/len(texts):.0%})"
        )
        
        return filtered
    
    def validate_sentence(self, sentence: str) -> Tuple[bool, str]:
        """
        Validate if a sentence meets preprocessing criteria.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not sentence or not isinstance(sentence, str):
            return False, "Empty or non-string input"
        
        # Check length
        if len(sentence) < self.min_sentence_length:
            return False, f"Too short (min {self.min_sentence_length} chars)"
        
        if len(sentence) > self.max_sentence_length:
            return False, f"Too long (max {self.max_sentence_length} chars)"
        
        # Check if it's mostly whitespace
        if sentence.strip() == "":
            return False, "Only whitespace"
        
        # Check if it's mostly punctuation/numbers (if those are removed)
        cleaned = self.preprocess_text(sentence)
        if len(cleaned) < self.min_sentence_length:
            return False, "Mostly punctuation/numbers after cleaning"
        
        return True, "Valid"


# Singleton instance for easy import
_default_preprocessor: Optional[TextPreprocessor] = None


def get_preprocessor() -> TextPreprocessor:
    """
    Get or create the default text preprocessor instance.
    
    Returns:
        Shared TextPreprocessor instance
    """
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = TextPreprocessor()
        logger.info("Created default text preprocessor instance")
    
    return _default_preprocessor


# Convenience functions for common operations
def clean_text(text: str) -> str:
    """
    Convenience function to clean text using default preprocessor.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    preprocessor = get_preprocessor()
    return preprocessor.preprocess_text(text)


def split_text_into_sentences(text: str) -> List[str]:
    """
    Convenience function to split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    preprocessor = get_preprocessor()
    return preprocessor.split_into_sentences(text)


def extract_key_terms_from_text(text: str, top_n: int = 10) -> List[str]:
    """
    Convenience function to extract key terms from text.
    
    Args:
        text: Input text
        top_n: Number of top terms to return
        
    Returns:
        List of key terms
    """
    preprocessor = get_preprocessor()
    return preprocessor.extract_key_terms(text, top_n=top_n)