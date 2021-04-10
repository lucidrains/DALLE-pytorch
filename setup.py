from setuptools import setup, find_packages

setup(
  name = 'perugia',
  packages = find_packages(),
  include_package_data = True,
  version = '0.9.6',
  license='MIT',
  description = 'perugia - generate images from text with machine learning (gpu needed)',
  author = 'Sam Sepiol',
  author_email = 'samsepi0l@fastmail.com',
  url = 'https://github.com/afiaka87/perugia',
  keywords = [
    'artificial intelligence',
    'text-to-image'
  ],
  install_requires=[
    'axial_positional_embedding',
    'DALL-E',
    'einops>=0.3',
    'ftfy',
    'pillow',
    'regex',
    'taming-transformers',
    'torch>=1.6',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
