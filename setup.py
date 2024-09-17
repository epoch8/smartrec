import setuptools

setuptools.setup(
    name="smartrec",
    version="0.0.1",
    include_package_data=True,
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "mypy",
        ],
    },
)