"""
Created on 16.04.2024
@author: Alexis MELOT
@director: Sean WOOD
@co-director: Fabien ALIBART, Pierre YGER, Yannick COFFINIER
University: U. Sherbrooke, U. Lille
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

setup(
    name="sparsesorter", 
    version="1.17",
    packages=find_packages(),
    author="Alexis MELOT",
    author_email="alexis.melot.etu@univ-lille.fr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", #TODO: add url
    python_requires=">=3.10",
    # install_requires=requirements,
    license="MIT",
    )

