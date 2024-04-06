"""BayesABTest package configuration."""

from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='BayesABTest',
    version='1.0.9',
    author='Baker Moran',
    author_email='bamoran99@gmail.com',
    license='MIT',
    description='A package for running AB tests in a Bayesian framework.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['AB Test', 'Bayes', 'Bayesian Statistics'],
    url='https://github.com/bakermoran/BayesABTest',
    download_url='https://github.com/bakermoran/BayesABTest/archive/v1.0.9-alpha.tar.gz',
    packages=['BayesABTest'],
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0,<1.24.0',
        'pandas<1.5.0',
        'matplotlib<4.0.0',
        'seaborn<0.11.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)
