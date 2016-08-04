#! /usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import brian2lems 

requirements = [
    'brian2', 'lems'
]

setup(  name='brian2lems',
        version = '0.1',
        description='NeuroML/LEMS exporter from Brian2 code',
        author='Dominik Krzeminski',
        packages=['brian2lems'],
        package_data={'brian2lems': ['*.xml']},
        include_package_data=True,
     )
