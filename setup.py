# The setup.py file is essential for making your project installable as a package, which enables you to import specific modules and components using import statements (like from mlProject.components import dataingestion).

# provides tools for setting up and distributing Python packages
import setuptools

with  open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()
    
__version__ = "0.0.0"

REPO_NAME = "End-To-End-Machine-Learning-Project-with-MlFlow"
AUTHOR_USERNAME = "InsightfulCoderDivine"
SRC_REPO = "mlproject"
AUTHOR_EMAIL = "divinenwadigo06@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USERNAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}/issues"
    },
    # Indicates that the src directory contains the main package code.
    package_dir={"": "src"},
    # Automatically detects all packages (folders with __init__.py files) inside the src directory and makes them part of the distribution.
    packages=setuptools.find_packages(where="src")
)