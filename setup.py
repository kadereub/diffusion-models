from setuptools import setup

setup(
    name='diffusion',
    version='0.1.0',
    author='Reuben Kade',
    author_email='',
    packages=['diffusion', 'diffusion.test'],
    # scripts=[''],
    url='',
    license='',
    description='A collection of simple diffusion models to simulate asset prices.',
    long_description=open('README.md').read(),
    install_requires=[
       "numpy >= 1.22.0",
       "numba >= 0.56.0",
    ],
)
