#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup

# setup(
#     name='virtual_home',
#     version='0.0.1',
#     description='virtual_home_0.2.0',
#     packages=['virtual_home'],
#     package_dir={},

#     install_requires=[
#         'certifi==2019.3.9',
#         'chardet==3.0.4',
#         'idna==2.8',
#         'numpy>=1.16.2',
#         'opencv-python==4.0.0.21',
#         'pillow>=7.1.0',
#         'requests==2.21.0',
#         'termcolor==1.1.0',
#         'tqdm==4.31.1',
#         'urllib3>=1.24.2',
#         'plotly==3.10',
#         'networkx==2.3',
#     ],

#     license='MIT',
# )

setup(
    name='virtual_home',
    version='0.0.1',
    description='virtual_home_0.2.0',
    packages=['virtual_home'],
    package_dir={},
    install_requires=
    [
        'certifi', 'chardet', 'idna', 'numpy', 'opencv-python', 
        'Pillow', 'requests', 'termcolor', 'tqdm', 'urllib3', 
        'plotly', 'networkx'
    ]#And any other dependencies required
)

