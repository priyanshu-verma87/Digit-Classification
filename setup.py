from setuptools import setup
setup(
    name="digit-classifier",
    python_requires=">=3.10,<3.11",  # Hard lock to Python 3.10
    install_requires=["tensorflow==2.12.0", "streamlit"],
)
