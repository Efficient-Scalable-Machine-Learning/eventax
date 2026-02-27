import io
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eventax",
    version="0.1.0",
    author="Lukas König",
    author_email="lukmkoenig@gmail.com",
    description=(
        "A Diffrax-based toolkit for continuous-time spiking neural networks "
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LuggiStruggi/EventPropJax",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "jax>=0.4.0",
        "diffrax>=0.6.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
