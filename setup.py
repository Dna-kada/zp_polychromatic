from setuptools import setup, find_packages

setup(
    name="zpxrm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "mpi4py",
        # etc.
    ],
    python_requires=">=3.10",
)