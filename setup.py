from setuptools import find_packages, setup

from os import path

ROOT = path.abspath(path.dirname(__file__))

with open(path.join(ROOT, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="text2graph-api",
    version="0.1.0",
    description="Use this library to transform raw text into differents graph representations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="PLN-disca-iimas",
    author_email="andric.valdez@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'sklearn',
        'unidecode',
        'nltk',
        'gensim',
        'matplotlib',
        'networkx',
        'pandas',
        'numpy',
        'emoji',
        'sphinx'
    ]
)