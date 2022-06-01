import pathlib
from setuptools import setup


HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='jajapy',
    long_description=README,
    packages=['jajapy'],
    install_requires=['numpy', 'scipy'],
    long_description_content_type="text/markdown",
    version='0.0.1',
    url="",
    description='''''',
    author='Raphaël Reynouard',
    author_email="raphal20@ru.is",
    license='MIT',
)