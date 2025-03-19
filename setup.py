# -*- coding: utf-8 -*-
# @Time    : 2024/10/13
# @Author  : wenshao
# @Project : wilor_keypoints
# @FileName: setup.py

from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Separate GitHub dependencies
github_deps = [req for req in requirements if req.startswith('git+')]
regular_deps = [req for req in requirements if not req.startswith('git+')]

setup(
    name='wilor_keypoints',
    version='1.0',
    description='WiLoR python package',
    packages=find_packages(),
    install_requires=regular_deps,
    dependency_links=github_deps,
    python_requires='>=3.8',
    data_files=[]
)
