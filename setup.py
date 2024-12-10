import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the content of the requirements.txt file	
with open('requirements.txt', 'r', encoding='utf-8') as f:	
    requirements = f.read().splitlines()

setuptools.setup(
    name="syncode",
    version="0.1",
    author="Alex Marzban",
    author_email="marza@bu.edu",
    description="This package provides the tool for grammar augmented LLM generation.",
    url="https://github.com/shubhamugare/syncode",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=requirements,
)