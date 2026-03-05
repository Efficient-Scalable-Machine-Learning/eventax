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
        "A Diffrax-based framework for continuous-time spiking neural networks "
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Efficient-Scalable-Machine-Learning/eventax",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "jax>=0.9.1",
        "diffrax>=0.7.2",
        "equinox>=0.13.5",
    ],
    python_requires=">=3.11",
)
