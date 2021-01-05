#!/usr/bin/env python

from setuptools import setup

setup(
    name="neural-net-from-scratch",
    version="1.0.0",
    author="Ron Sabag",
    author_email="ronsabag135@gmail.com",
    url="https://github.com/ronsabag/neural-net",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Neural network implementation from scratch",
    long_description=open("README.md").read(),
    packages=["neural_net_from_scratch"],
    package_data={"test_data": ["test/*.txt"]},
    include_package_data=True,
    install_requires=[],
)
