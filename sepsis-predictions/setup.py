from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
  name='predictions_sepsis',
  version='0.0.1',
  author='@Margo78, @akp1n',
  author_email='timtimk30@yandex.ru',
  description='Module for sepsis_predictions',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/sslavian812/sepsis-predictions',
  packages=find_packages(),
  install_requires=['requests>=2.25.1'],
  classifiers=[
    'Programming Language :: Python :: 3.11',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='example python',
  project_urls={
    'Documentation': 'link'
  },
  python_requires='>=3.7'
)