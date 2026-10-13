#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=[
        "lightning",
        "hydra-core",
        "numpy",
        "scikit-learn>=1.5,<2",
        "optuna>=3.6,<5",
        "polars",
    ],
    packages=find_packages(),
    extras_require={"huggingface": ["transformers>=4.48.1,<5", "datasets>=4.0.0,<5"]},
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
            "research_command = src.research_cli:main",
        ]
    },
)
