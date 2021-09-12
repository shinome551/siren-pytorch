from setuptools import setup, find_packages

setup(
  name = 'siren-pytorch',
  packages = find_packages(),
  version = '0.1.6',
  license='MIT',
  description = 'Implicit Neural Representations with Periodic Activation Functions',
  author = 'Ryota Higashi',
  author_email = 'reinshinome@gmail.com',
  url = 'https://github.com/shinome551/siren-pytorch',
  keywords = ['artificial intelligence', 'deep learning'],
  install_requires=[
      'einops',
      'torch'
  ],
)