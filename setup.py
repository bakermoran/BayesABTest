"""BayesABTest package configuration."""

from setuptools import setup

setup(
    name='BayesABTest',
    version='1.0.2',
    author="Baker Moran",
    author_email="bamoran99@gmail.com",
    description="A package for running AB tests in a Bayesian framework.",
    url="https://github.com/bakermoran/BayesABTest",
    packages=['BayesABTest'],
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16.5',
        'pandas>=0.24.2',
        'pymc3>=3.7',
        'matplotlib>=1.3.1',
        'seaborn>=0.9.0'
    ],
)
