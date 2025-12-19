from setuptools import setup, find_packages

setup(
    name="text_analysis_service",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.2",
        "vaderSentiment>=3.3.2",
    ],
    python_requires=">=3.8",
)
