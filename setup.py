from setuptools import setup

setup(
    name='BNN',
    version='1.0',
    description='Bayesian Neural Networks to estimate chlorophyll-a concentration from Sentinel-3 OLCI and Sentinel-2 MSI in oligotrophic and mesotrophic lakes',
    author='Mortimer Werther',
    author_email='mortimer.werther@eawag.ch',
    url='https://github.com/mowerther/BNN_2022',
    package_dir={'BNN_2022':''},
    packages=['BNN_2022'],
)