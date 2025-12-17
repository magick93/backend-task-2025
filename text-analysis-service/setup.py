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
    ],
    python_requires=">=3.8",
)