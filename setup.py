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
	name='jaxstronomy',
	version='0.0.0',
	description='Strong lensing package',
	long_description=readme,
	author='Sebastian Wagner-Carena',
	author_email='sebaswagner@outlook.com',
	url='https://github.com/swagnercarena/jaxstronomy',
	packages=find_packages(PACKAGE_PATH),
	package_dir={'jaxstronomy': 'jaxstronomy'},
	include_package_data=True,
	install_requires=required_packages,
	license='MIT',
	zip_safe=False
)
