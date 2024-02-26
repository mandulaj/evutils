from setuptools import setup, find_packages

setup(
    name='ev_utils',
    version='0.1',
    packages=find_packages('src'),  # This tells setuptools to look for packages in the `src` directory
    package_dir={'': 'src'},  # This maps the package names to the `src` directory
)