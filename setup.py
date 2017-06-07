# setup.py
from setuptools import setup, find_packages

setup(name='example1',
  version='0.1',
  packages=find_packages(),
  description='example to run keras on gcloud ml-engine',
  author='Nabila63Ahmed',
  author_email='nabila.ahmed63@gmail.com',
  license='MIT',
  install_requires=[
      'keras',
      'h5py'
  ],
  zip_safe=False)

