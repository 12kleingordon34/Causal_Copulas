from setuptools import find_packages, setup

setup(
    name='frugalCopyla',
    version='0.0.2',
    description='tbc',
    author='Dan Manela',
    author_email='tbc',
    install_requires=[
        'jax==0.3.24',
        'jaxlib==0.3.24',
        'numpy==1.23.4',
        'numpyro==0.10.1',
        'patsy==0.5.3'
    ],
    packages=find_packages(
        where='frugalCopyla',
        include=['model', 'copula_functions']
    )
)
