# Design Notes

This directory contains high-level design documentation and decision records for the Text Analysis Service.

## Contents

1.  [**Architecture & Infrastructure**](./01_architecture_infrastructure.md)
    *   Serverless design decisions.
    *   API Gateway & Monolithic Lambda pattern.
    *   Observability setup.

2.  [**ML Pipeline Strategy**](./02_ml_pipeline_strategy.md)
    *   Pipeline orchestration.
    *   Model selection:
        *   **`sentence-transformers`**: Pre-trained deep learning models for high-quality sentence embeddings (chosen for balance of speed and semantic accuracy).
        *   **DBSCAN**: Density-based clustering algorithm (chosen because it doesn't require specifying the number of clusters K upfront).
        *   **VADER**: Lexicon and rule-based sentiment analysis tool (chosen for speed and effectiveness on social media/short text without needing training).
    *   Clustering and Insight generation.

3.  [**Quality & Testing**](./03_quality_and_testing.md)
    *   Testing pyramid (Unit vs. Integration).
    *   Local testing with `sam local`.
    *   Project structure and Dependency Injection.
