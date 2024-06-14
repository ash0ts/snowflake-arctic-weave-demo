from setuptools import find_packages, setup


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, "r") as file:
        lines = file.readlines()
        # Filter out comments and empty lines
        requirements = [
            line.strip() for line in lines if line.strip() and not line.startswith("#")
        ]
    return requirements


setup(
    name="weave_example_demo",
    version="0.1",
    packages=find_packages(),
    # install_requires=parse_requirements('requirements.txt'),
    install_requires=[],
)
