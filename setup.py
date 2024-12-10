import setuptools
# Read the content of the requirements.txt file	
with open('requirements.txt', 'r', encoding='utf-8') as f:	
    requirements = f.read().splitlines()

setuptools.setup(
    name="syncode2",
    version="1",
    author="Alex Marzban",
    author_email="marza@bu.edu",
    description="This package provides the tool for grammar augmented LLM generation.",
    url="https://github.com/marzbana/GARS/syncode2",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=requirements,
)