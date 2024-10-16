# -*- coding: utf-8 -*-
# @Time    : 2024/10/13
# @Author  : wenshao
# @Project : WiLoR-mini
# @FileName: setup.py

from setuptools import setup

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='wilor_mini',
    version='1.0',
    description='WiLoR python package',
    packages=[
        'wilor_mini',
        'wilor_mini.models',
        'wilor_mini.pipelines',
        'wilor_mini.utils'
    ],
    install_requires=requirements,
    python_requires='>=3.8',
    data_files=[]
)
