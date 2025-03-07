from setuptools import setup, find_packages

setup(
    name="kserve_torch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["kserve>=0.10.0", "torch>=1.13.0"],
)
