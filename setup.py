from setuptools import find_packages
from setuptools import setup

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='Finnet Autoencoder Anomaly Detection',
    # ADD THIS SECTION:
    install_requires=[
        'google-cloud-bigquery>=3.20.0', # Forces an upgrade to a fixed version
        'google-api-core>=2.17.0',
        'google-cloud-storage',
        'pandas', 
        'numpy',
        'matplotlib',
        'seaborn',
        'fastparquet'
    ]
)