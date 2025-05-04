from setuptools import setup, find_packages

setup(
    name="gsmp_vglm",
    version="0.1.0",
    description="A test library for GSMP Vector Generalized Linear Models",
    author="Erick Luerken",
    url="https://github.com/Schlagmichtot/gsmp_vglm",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.1.0",
        "scipy>=1.15.0",
        "matplotlib>=3.10.0",
        "pandas>=2.2.0",
	"scikit-optimize>=0.10.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
)