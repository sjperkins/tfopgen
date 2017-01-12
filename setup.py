import os

from setuptools import setup, find_packages

from install.versioning import maintain_version

def readme(filename):
  return open(filename).read()

setup(name='tfopgen',
      description='Generates tensorflow custom operator boilerplate',
      long_description=readme('README.rst'),
      url='',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      author='Simon Perkins',
      author_email='simon.perkins@gmail.com',
      license='',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'jinja2 >= 2.8.0',
          'numpy >= 1.12.0',
          'ruamel.yaml >= 0.13.7'
      ],
      scripts=[os.path.join('tfopgen', 'bin', 'tfopgen')],
      version=maintain_version(os.path.join('tfopgen', 'version.py')),
      zip_safe=True)
