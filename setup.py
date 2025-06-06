from setuptools import setup, find_packages

setup(
    name="correlated_noise_mechanism",
    version="0.1.0",
    author="Ashish Srivastava",
    author_email="ashish.srivastava1919@gmail.com",
    description="Implementation of correlated noise mechanism with streaming and multi epoch settings to enable differentially private deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/grim-hitman0XX/correlated_noise_mechanism",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.26.4",
        "torch >= 2.3.0+cu121",
        "torchvision >= 0.18.0+cu121",
        "opacus >= 1.5.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)