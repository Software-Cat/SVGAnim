from setuptools import setup


README = open("README.md").read()

setup(
    name="svganim",
    version="1.0.0",
    author="Software Cat",
    long_description=README,
    license="MIT",
    packages=["svganim"],
)
