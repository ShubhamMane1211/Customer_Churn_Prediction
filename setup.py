"""Compatibility entry point for older versions of pip."""

from setuptools import find_packages, setup


setup(
    name="customer-churn-predictor",
    version="0.1.0",
    description="Bank customer churn prediction package and Streamlit interface.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["joblib>=1.3", "pandas>=2.0", "scikit-learn>=1.3", "streamlit>=1.30"],
    extras_require={"dev": ["pytest>=8.0"]},
    entry_points={"console_scripts": ["churn-predict=churn_predictor.inference:main"]},
    python_requires=">=3.10",
)
