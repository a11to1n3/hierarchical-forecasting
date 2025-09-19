from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hierarchical-forecasting",
    version="0.1.0",
    description="Hierarchical Forecasting with Combinatorial Complex Message Passing Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/hierarchical-forecasting",
    packages=find_packages(),
    package_dir={"hierarchical_forecasting": "hierarchical_forecasting"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hierarchical-train=scripts.train:main",
            "hierarchical-evaluate=scripts.evaluate:main",
            "hierarchical-visualize=scripts.visualize:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="hierarchical forecasting, neural networks, combinatorial topology, time series",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hierarchical-forecasting/issues",
        "Source": "https://github.com/yourusername/hierarchical-forecasting",
        "Documentation": "https://hierarchical-forecasting.readthedocs.io/",
    },
)
