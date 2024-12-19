from setuptools import setup, find_packages

setup(
    name='mgds',
    version='0.1',
    description='Modular Pytorch DataSet implementation',
    author='Nerogar',
    url='https://github.com/Nerogar/stable-diffusion-dataloader/',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)
