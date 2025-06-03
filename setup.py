from setuptools import setup, find_packages

setup(
    name="simlib",
    version="1.0.0",
    author="SimLib Team",
    description="A comprehensive simulation library based on Sheldon Ross's Simulation textbook",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "networkx>=2.5",
        "seaborn>=0.11.0",
        "PyQt6>=6.2.0",  # For GUI
        "plotly>=5.0.0",  # Interactive plots
        "tqdm>=4.60.0",   # Progress bars
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "simlib-gui=simlib.ui.main:main",
        ],
    },
)