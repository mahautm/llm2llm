from setuptools import find_packages, setup

setup(
    name="llm",
    version="0.0",
    description="train a llm to talk to another llm and combine information from both to answer a question",
    author="Matéo Mahaut",
    author_email="mateo.mahaut@upf.edu",
    packages=find_packages(),  # same as name
    install_requires=[
        "torch",
        "transformers",
        "pandas",
    ],  # external packages as dependencies
)
