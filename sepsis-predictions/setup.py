from setuptools import setup, find_packages

def readme():
    with open('README.md', 'r') as f:
        return f.read()

setup(
    name='predictions_sepsis',
    version='1.0.3',
    author='@Margo78, @akp1n',
    author_email='timtimk30@yandex.ru',
    description='Module for sepsis predictions',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/sslavian812/sepsis-predictions.git',
    packages=find_packages(),
    install_requires=[
        'requests>=2.25.1',
        'pandas>=1.3.3',
        'tqdm>=4.62.3',
        'numpy>=1.21.2',
        'scikit-learn>=0.24.2',
        'imbalanced-learn>=0.8.0',
        'pytorch-tabnet>=3.1.1',
        'torch>=1.9.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords='sepsis, predictions, python',
    python_requires='>=3.7'
)