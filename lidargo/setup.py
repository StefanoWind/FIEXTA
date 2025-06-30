from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lidargo",
    version="0.1.1",
    author="Nicholas Hamilton",
    author_email="nicholas.hamilton@nrel.gov",
    description="A package for LIDAR data processing and analysis",
    long_description=long_description,
    long_description_content_type="LIDARGO (LIDAR General Operator): Standardization, quality control, and statistics of scanning lidar data.",
    url="hhttps://github.com/nREL/fiexta/LIDARGO",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "netCDF4>=1.5.7",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9.0",
            "black>=21.0",
        ],
    },
)
