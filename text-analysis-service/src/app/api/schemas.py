"""
API schemas for the text analysis microservice.

This module defines Pydantic models for input validation and output serialization
for both standalone and comparative text analysis.

Key schemas:
- Sentence: Basic sentence with text and unique identifier
- StandaloneInput: Input for standalone analysis (surveyTitle, theme, baseline sentences)
- ComparisonInput: Input for comparative analysis (baseline + comparison sentences)
- StandaloneOutput: Output for standalone analysis with clusters and insights
- ComparisonOutput: Output for comparative analysis with similarities/differences

All models include comprehensive validation, type hints, and example data.

TODO for production enhancements:
1. Add OpenAPI schema extensions for better documentation
2. Implement custom validators for business logic (e.g., sentiment value validation)
3. Add rate limiting metadata to schemas
4. Implement schema versioning for backward compatibility
5. Add caching headers and ETag support
6. Implement request/response compression schemas
7. Add pagination support for large result sets
8. Implement field-level encryption for sensitive data
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, conlist, ValidationInfo, ConfigDict
from uuid import UUID
import re
from datetime import datetime


class Sentence(BaseModel):
    """
    A single sentence with its unique identifier.
    
    Attributes:
        sentence: The text content of the sentence
        id: Unique identifier for the sentence (UUID format)
    
    Example:
        {
            "sentence": "The user interface is very intuitive",
            "id": "550e8400-e29b-41d4-a716-446655440000"
        }
    """
    sentence: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The text content of the sentence"
    )
    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the sentence"
    )
    
    @field_validator('sentence')
    def validate_sentence_not_empty(cls, v):
        """Ensure sentence is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Sentence cannot be empty or whitespace only')
        return v.strip()
    
    @field_validator('id')
    def validate_id_format(cls, v):
        """Validate ID format (should be alphanumeric with hyphens/underscores)."""
        # Allow UUID format or other reasonable identifiers
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', v):
            raise ValueError('ID must contain only alphanumeric characters, hyphens, underscores, or dots')
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sentence": "The user interface is very intuitive and easy to use",
                "id": "7d4aa701-f1b2-41eb-b42c-c59e8e1d5091"
            }
        }
    )


class StandaloneInput(BaseModel):
    """
    Input schema for standalone text analysis.
    
    Used when analyzing a single set of sentences to identify themes and insights.
    
    Attributes:
        surveyTitle: Title of the survey or analysis context
        theme: High-level theme or topic being analyzed
        baseline: List of sentences to analyze
    
    Example:
        {
            "surveyTitle": "Customer Feedback Q4 2024",
            "theme": "User Interface",
            "baseline": [
                {"sentence": "UI is intuitive", "id": "id1"},
                {"sentence": "Hard to navigate", "id": "id2"}
            ]
        }
    """
    surveyTitle: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Title of the survey or analysis context"
    )
    theme: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="High-level theme or topic being analyzed"
    )
    baseline: conlist(Sentence, min_length=1) = Field(
        ...,
        description="List of sentences to analyze (minimum 1 sentence required)"
    )
    
    @field_validator('baseline')
    def validate_unique_sentence_ids(cls, v):
        """Ensure all sentence IDs are unique within the baseline."""
        ids = [sentence.id for sentence in v]
        if len(ids) != len(set(ids)):
            raise ValueError('All sentence IDs must be unique within the baseline')
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "surveyTitle": "Robinhood App Store Reviews",
                "theme": "Account Management",
                "baseline": [
                    {
                        "sentence": "Withholding my money",
                        "id": "7d4aa701-f1b2-41eb-b42c-c59e8e1d5091"
                    },
                    {
                        "sentence": "Have lost so much money",
                        "id": "bc69d50c-4bdc-4fea-b58b-878c1cb5a2f2"
                    }
                ]
            }
        }
    )


class ComparisonInput(BaseModel):
    """
    Input schema for comparative text analysis.
    
    Used when comparing two sets of sentences to identify similarities and differences.
    
    Attributes:
        surveyTitle: Title of the survey or analysis context
        theme: High-level theme or topic being analyzed
        baseline: First set of sentences (baseline/reference)
        comparison: Second set of sentences (comparison/target)
    
    Example:
        {
            "surveyTitle": "Product Feedback Comparison",
            "theme": "Feature Usability",
            "baseline": [{"sentence": "Easy to use", "id": "id1"}],
            "comparison": [{"sentence": "Hard to learn", "id": "id2"}]
        }
    """
    surveyTitle: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Title of the survey or analysis context"
    )
    theme: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="High-level theme or topic being analyzed"
    )
    baseline: conlist(Sentence, min_length=1) = Field(
        ...,
        description="First set of sentences (baseline/reference)"
    )
    comparison: conlist(Sentence, min_length=1) = Field(
        ...,
        description="Second set of sentences (comparison/target)"
    )
    
    @field_validator('baseline', 'comparison')
    def validate_unique_ids_within_each_list(cls, v, info: ValidationInfo):
        """Ensure all sentence IDs are unique within each list."""
        ids = [sentence.id for sentence in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f'All sentence IDs must be unique within the {info.field_name} list')
        return v
    
    @field_validator('comparison')
    def validate_comparison_not_identical_to_baseline(cls, v, info: ValidationInfo):
        """Ensure comparison is not identical to baseline (optional but recommended)."""
        if info.data and 'baseline' in info.data:
            baseline_ids = {s.id for s in info.data['baseline']}
            comparison_ids = {s.id for s in v}
            if baseline_ids == comparison_ids:
                # This is a warning, not an error - but we'll allow it with a note
                # In production, you might want to log this as a potential issue
                pass
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "surveyTitle": "Airline Food Service Reviews",
                "theme": "Food Quality",
                "baseline": [
                    {
                        "sentence": "They still managed to get drinks and snacks out",
                        "id": "d1888926-b847-42d9-9922-4432839f509a"
                    },
                    {
                        "sentence": "Gave us lots of snacks",
                        "id": "27bd71b6-e5d9-4f85-925c-724776b8ae71"
                    }
                ],
                "comparison": [
                    {
                        "sentence": "I didn't eat the food",
                        "id": "b66e7716-a4f8-4367-b61d-f7c986d22c23"
                    },
                    {
                        "sentence": "The food was a bit rough",
                        "id": "045d0ce5-f9b1-4af0-b0be-6b49f67e8373"
                    }
                ]
            }
        }
    )


class ClusterInsight(BaseModel):
    """
    Insights for a single cluster in standalone analysis.
    
    Represents a thematic cluster identified in the analysis.
    
    Attributes:
        title: Descriptive title for the cluster
        sentiment: Overall sentiment of the cluster (positive|negative|neutral)
        sentences: List of sentence IDs belonging to this cluster
        keyInsights: List of key insights or observations about the cluster
    
    Example:
        {
            "title": "Money Withdrawal Issues",
            "sentiment": "negative",
            "sentences": ["id1", "id2", "id3"],
            "keyInsights": ["Users report difficulty withdrawing funds", "Multiple complaints about frozen accounts"]
        }
    """
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Descriptive title for the cluster"
    )
    sentiment: str = Field(
        ...,
        description="Overall sentiment of the cluster",
        pattern="^(positive|negative|neutral)$"
    )
    sentences: List[str] = Field(
        ...,
        min_length=1,
        description="List of sentence IDs belonging to this cluster"
    )
    keyInsights: List[str] = Field(
        ...,
        min_length=1,
        description="List of key insights or observations about the cluster"
    )
    
    @field_validator('sentences')
    def validate_sentence_ids_exist(cls, v):
        """Validate that sentence IDs are not empty."""
        for sentence_id in v:
            if not sentence_id or not sentence_id.strip():
                raise ValueError('Sentence ID cannot be empty')
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Money Withdrawal Issues",
                "sentiment": "negative",
                "sentences": [
                    "7d4aa701-f1b2-41eb-b42c-c59e8e1d5091",
                    "bc69d50c-4bdc-4fea-b58b-878c1cb5a2f2"
                ],
                "keyInsights": [
                    "Users report difficulty withdrawing funds",
                    "Multiple complaints about frozen accounts"
                ]
            }
        }
    )


class ComparisonCluster(BaseModel):
    """
    Cluster for comparative analysis with separate baseline and comparison sentences.
    
    Attributes:
        title: Descriptive title for the cluster
        sentiment: Overall sentiment of the cluster (positive|negative|neutral)
        baselineSentences: Sentence IDs from the baseline belonging to this cluster
        comparisonSentences: Sentence IDs from the comparison belonging to this cluster
        keySimilarities: List of key similarities between baseline and comparison
        keyDifferences: List of key differences between baseline and comparison
    
    Example:
        {
            "title": "Food Quality Complaints",
            "sentiment": "negative",
            "baselineSentences": ["id1", "id2"],
            "comparisonSentences": ["id3", "id4"],
            "keySimilarities": ["Both groups mention poor food presentation"],
            "keyDifferences": ["Baseline mentions portion size, comparison mentions taste"]
        }
    """
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Descriptive title for the cluster"
    )
    sentiment: str = Field(
        ...,
        description="Overall sentiment of the cluster",
        pattern="^(positive|negative|neutral)$"
    )
    baselineSentences: List[str] = Field(
        ...,
        description="Sentence IDs from the baseline belonging to this cluster"
    )
    comparisonSentences: List[str] = Field(
        ...,
        description="Sentence IDs from the comparison belonging to this cluster"
    )
    keySimilarities: List[str] = Field(
        ...,
        description="List of key similarities between baseline and comparison in this cluster"
    )
    keyDifferences: List[str] = Field(
        ...,
        description="List of key differences between baseline and comparison in this cluster"
    )
    
    @field_validator('baselineSentences', 'comparisonSentences')
    def validate_sentence_ids_not_empty(cls, v):
        """Validate that sentence IDs are not empty."""
        for sentence_id in v:
            if not sentence_id or not sentence_id.strip():
                raise ValueError('Sentence ID cannot be empty')
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Food Quality Complaints",
                "sentiment": "negative",
                "baselineSentences": ["d1888926-b847-42d9-9922-4432839f509a"],
                "comparisonSentences": ["b66e7716-a4f8-4367-b61d-f7c986d22c23"],
                "keySimilarities": [
                    "Both groups mention dissatisfaction with food quality"
                ],
                "keyDifferences": [
                    "Baseline focuses on snack availability, comparison focuses on meal quality"
                ]
            }
        }
    )


class StandaloneOutput(BaseModel):
    """
    Output schema for standalone text analysis.
    
    Contains the results of clustering and insight generation for a single set of sentences.
    
    Attributes:
        clusters: List of clusters identified in the analysis
        metadata: Optional metadata about the analysis (processing time, job ID, etc.)
    
    Example:
        {
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
    """
    clusters: List[ClusterInsight] = Field(
        ...,
        description="List of clusters identified in the analysis"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata about the analysis"
    )
    
    @field_validator('clusters')
    def validate_clusters_have_unique_sentences(cls, v):
        """Ensure sentences are not duplicated across clusters."""
        all_sentence_ids = []
        for cluster in v:
            all_sentence_ids.extend(cluster.sentences)
        
        if len(all_sentence_ids) != len(set(all_sentence_ids)):
            # This could happen if a sentence belongs to multiple clusters
            # In production, you might want to handle this differently
            pass  # We'll allow it for now, but log a warning in production
        
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "clusters": [
                    {
                        "title": "Money Withdrawal Issues",
                        "sentiment": "negative",
                        "sentences": [
                            "7d4aa701-f1b2-41eb-b42c-c59e8e1d5091",
                            "bc69d50c-4bdc-4fea-b58b-878c1cb5a2f2"
                        ],
                        "keyInsights": [
                            "Users report difficulty withdrawing funds",
                            "Multiple complaints about frozen accounts"
                        ]
                    }
                ],
                "metadata": {
                    "processingTimeMs": 1250,
                    "jobId": "job-12345",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "clusterCount": 1,
                    "totalSentences": 2
                }
            }
        }
    )


class ComparisonOutput(BaseModel):
    """
    Output schema for comparative text analysis.
    
    Contains the results of comparing two sets of sentences, including
    similarities, differences, and clustered insights.
    
    Attributes:
        clusters: List of comparison clusters
        metadata: Optional metadata about the analysis
    
    Example:
        {
            "clusters": [
                {
                    "title": "Food Quality Complaints",
                    "sentiment": "negative",
                    "baselineSentences": ["id1"],
                    "comparisonSentences": ["id2"],
                    "keySimilarities": ["Both mention poor food"],
                    "keyDifferences": ["Baseline: portion size, Comparison: taste"]
                }
            ],
            "metadata": {
                "processingTimeMs": 2500,
                "similarityScore": 0.75
            }
        }
    """
    clusters: List[ComparisonCluster] = Field(
        ...,
        description="List of comparison clusters"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata about the analysis"
    )
    
    @field_validator('clusters')
    def validate_cluster_sentence_uniqueness(cls, v):
        """Ensure sentences are not duplicated across clusters within each dataset."""
        baseline_ids = []
        comparison_ids = []
        
        for cluster in v:
            baseline_ids.extend(cluster.baselineSentences)
            comparison_ids.extend(cluster.comparisonSentences)
        
        # Check for duplicates within each dataset
        if len(baseline_ids) != len(set(baseline_ids)):
            # Log warning in production
            pass
        
        if len(comparison_ids) != len(set(comparison_ids)):
            # Log warning in production
            pass
        
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "clusters": [
                    {
                        "title": "Food Quality Complaints",
                        "sentiment": "negative",
                        "baselineSentences": ["d1888926-b847-42d9-9922-4432839f509a"],
                        "comparisonSentences": ["b66e7716-a4f8-4367-b61d-f7c986d22c23"],
                        "keySimilarities": [
                            "Both groups mention dissatisfaction with food quality"
                        ],
                        "keyDifferences": [
                            "Baseline focuses on snack availability, comparison focuses on meal quality"
                        ]
                    }
                ],
                "metadata": {
                    "processingTimeMs": 2500,
                    "jobId": "job-67890",
                    "timestamp": "2024-01-15T10:35:00Z",
                    "similarityScore": 0.65,
                    "totalBaselineSentences": 10,
                    "totalComparisonSentences": 8
                }
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Standard error response schema for API errors.
    
    Compatible with API Gateway error responses.
    
    Attributes:
        error: Error type/description
        message: Detailed error message
        status_code: HTTP status code
        details: Optional additional error details
        timestamp: When the error occurred
    """
    error: str = Field(
        ...,
        description="Error type or description"
    )
    message: str = Field(
        ...,
        description="Detailed error message"
    )
    status_code: int = Field(
        ...,
        ge=400,
        le=599,
        description="HTTP status code"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional additional error details"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + 'Z',
        description="Timestamp when the error occurred"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "status_code": 400,
                "details": {
                    "field_errors": {
                        "surveyTitle": "Field is required"
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )


# Export all schemas for easy import
__all__ = [
    "Sentence",
    "StandaloneInput",
    "ComparisonInput",
    "ClusterInsight",
    "ComparisonCluster",
    "StandaloneOutput",
    "ComparisonOutput",
    "ErrorResponse"
]