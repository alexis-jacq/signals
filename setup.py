 #!/usr/bin/env python
 # -*- coding: utf-8 -*-

import os
from distutils.core import setup
import glob

setup(name='signalLearner',
      version='0.0',
      license='ISC',
      description='RL agents learning actions from signals',
      author='Alexis Jacq',
      author_email='alexis.jacq@gmail.com',
      package_dir = {'': 'src'},
      packages=['signal_learner'],
      data_files=[('share/doc/', ['AUTHORS', 'LICENSE', 'README.md'])]
      )
