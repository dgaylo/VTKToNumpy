from setuptools import find_packages, setup

setup(
    name='vtktonumpy',
    packages=find_packages(include=['vtktonumpy']),
    version='dev',
    description='A python library to read VTK file and output as multidimensional numpy arrays',
    author='Declan Gaylo',
    install_requires=['vtk','numpy'],
)