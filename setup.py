from setuptools import find_packages, setup

setup(
    name="clv_pipeline",
    version="0.1.0",
    author="Sujato Dutta",
    author_email="sujatodutta0204@gmail.com",
    description="Customer Lifetime Value Prediction Pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
)
