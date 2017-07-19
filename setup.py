from setuptools import setup
setup(
  name = 'ddpg',
  packages = ['ddpg'], # this must be the same as the name above
  version = '0.1',
  description = 'Tensorflow implimentation of the DDPG algorithm',
  author = 'Max Ferguson',
  author_email = 'maxkferg@gmail.com',
  url = 'https://github.com/maxkferg/DDPG', # use the URL to the github repo
  download_url = 'https://github.com/maxkferg/DDPG/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['deep deterministic policy gradient', 'ddpg', 'machine learning'], # arbitrary keywords
  classifiers = [],
  install_requires=[
    'tensorflow',
    'matplotlib',
    'numpy'
  ],
)