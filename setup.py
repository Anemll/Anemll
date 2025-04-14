from setuptools import setup, find_packages

setup(
    name="anemll",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "coremltools>=8.2",    # Required for Apple Neural Engine conversion
        "numpy>=1.24.0",       # Needed for tensor operations
        "tqdm>=4.66.0",        # Progress bars for large conversions
        "transformers>=4.36.0" # HuggingFace Transformers support
    ],
    extras_require={
        "dev": [
            "black>=23.12.0",   # Code formatting
            "flake8>=7.0.0",    # Linting
            "pytest>=7.4.0",    # Testing
            "pytest-cov>=4.1.0" # Test coverage
        ]
    },
    description="Open-source pipeline for accelerating LLMs on Apple Neural Engine (ANE)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ANEMLL Team",
    author_email="realanemll@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha"
    ],
    python_requires=">=3.9",
    project_urls={
        "Homepage": "https://anemll.com",
        "Documentation": "https://anemll.com/docs",
        "Repository": "https://github.com/anemll/anemll",
        "Bug Tracker": "https://github.com/anemll/anemll/issues",
        "HuggingFace": "https://huggingface.co/anemll",
        "Twitter": "https://x.com/anemll"
    }
)
