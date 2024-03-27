#!/usr/bin/env python
try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

from setuptools import find_packages
import os

readme = open("README.rst").read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

required_packages = open('requirements.txt').read().splitlines()

setup(
	name='paltax',
	version='0.0.0',
	description='Strong lensing package using jax',
	long_description=readme,
	author='Sebastian Wagner-Carena',
	author_email='sebaswagner@outlook.com',
	url='https://github.com/swagnercarena/paltax',
	packages=find_packages(PACKAGE_PATH),
	package_dir={'paltax': 'paltax'},
	include_package_data=True,
	install_requires=required_packages,
	license='Apache2.0',
	zip_safe=False
)
