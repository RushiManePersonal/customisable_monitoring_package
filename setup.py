from setuptools import setup, find_packages

setup(
    name='monitoring',
    version='0.1.0',
    description='A package for data drift detection with multiple statistical tests',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'plotly',
        'kaleido'
    ],
    python_requires='>=3.7',
    package_data={
        'monitoring': [
            'drift_report_template.html',
            'drift_report_assets.js',
            '*.html',
            '*.js',
        ],
    },
    include_package_data=True,
)
