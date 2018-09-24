import os
from pathlib import Path
os.chdir(Path(Path(__file__).parent, '..').as_posix())
from setuptools import setup

setup(name='hfnet',
      version="0.0",
      packages=['hfnet'])
