from setuptools import setup, find_packages

setup(
    name='holograpy',
    version='0.1',
    entry_points={"console_scripts": ["holograpy=holograpy.__main__:main"]},
    packages=find_packages())