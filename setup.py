import pathlib
from setuptools import setup, find_packages
from codecs import open

HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(HERE / "README.md", encoding='utf-8') as f:
	long_description = f.read()
setup(
    name='jajapy',
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    long_description_content_type="text/markdown",
    version='0.5.1',
    url="",
    description='Baum-Welch for all kind of Markov model',
    author='Raphaël Reynouard',
    author_email="raphal20@ru.is",
    license='MIT',
	classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    include_package_data=True
)