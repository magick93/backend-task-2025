# ML Pipeline Strategy

## Overview
The ML pipeline is the core of the application, designed to transform raw text into actionable insights through a series of modular steps: Preprocessing, Embedding, Clustering, and Sentiment Analysis.

## Core Pipeline Strategy

### Pipeline Pattern
*   **Design**: Implemented using an `Orchestrator` pattern (`PipelineOrchestrator`).
*   **Extensibility**: The pipeline is constructed using dependency injection. Services (Embedding, Clustering, etc.) are injected into the orchestrator, making it easy to swap implementations or add new steps without rewriting core logic.
*   **Concurrency**: Uses `ThreadPoolExecutor` to run independent tasks (like Sentiment Analysis and Embedding generation) in parallel, reducing overall latency.

### Embedding Strategy
*   **Library**: `sentence-transformers` (Hugging Face).
*   **Model**: `all-MiniLM-L6-v2`.
*   **Decision**: This model offers the best balance of speed and quality for serverless environments. Larger models would increase cold start times and memory usage significantly.
*   **Optimization**: An in-memory `_embedding_cache` is implemented to reuse embeddings for identical inputs within the same Lambda warm execution context.

### Clustering Strategy
*   **Algorithm**: **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).
*   **Why DBSCAN?**: Unlike K-Means, DBSCAN does not require specifying the number of clusters ($K$) beforehand. This is crucial for dynamic text analysis where the number of topics is unknown.
*   **Noise Handling**: DBSCAN explicitly identifies "noise" (outliers), preventing irrelevant data from skewing topic clusters.

### Sentiment Analysis
*   **Library**: **VADER** (`vaderSentiment`).
*   **Decision**: Rule-based sentiment analysis was chosen over deep learning models (like BERT) for speed and simplicity. VADER is exceptionally fast and performs well on social media and short text, fitting the serverless constraints.

### Insight Generation
*   **Technique**: Frequency analysis and TF-IDF principles.
*   **Output**: Generates human-readable summaries by identifying the most representative words and phrases within clusters.

### Comparative Analysis
*   **Feature**: The pipeline supports a `process_comparison` mode.
*   **Logic**: It processes a "baseline" dataset and a "comparison" dataset to identify shifts in sentiment and topic distribution.
