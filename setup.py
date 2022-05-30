from setuptools import setup, find_packages
exec(open('dalle_pytorch/version.py').read())

setup(
  name = 'dalle-pytorch',
  packages = find_packages(),
  include_package_data = True,
  version = __version__,
  license='MIT',
  description = 'DALL-E - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/dalle-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'text-to-image'
  ],
  install_requires=[
    'axial_positional_embedding',
    'DALL-E',
    'einops>=0.3.2',
    'ftfy',
    'packaging',
    'pillow',
    'regex',
    'rotary-embedding-torch',
    'taming-transformers-rom1504',
    'tokenizers',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'tqdm',
    'youtokentome',
    'WebDataset'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
