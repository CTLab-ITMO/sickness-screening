from setuptools import setup, find_packages

def readme():
    with open('README.md', 'r') as f:
        return f.read()

setup(
    name='predictions_sepsis',
    version='0.0.6',
    author='@Margo78, @akp1n',
    author_email='timtimk30@yandex.ru',
    description='Module for sepsis predictions',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/sslavian812/sepsis-predictions.git',
    packages=find_packages(),
    install_requires=[
        'requests>=2.25.1',
        'pandas>=1.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords='sepsis, predictions, python',
    python_requires='>=3.7'
)