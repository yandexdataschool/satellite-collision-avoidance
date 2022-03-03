from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()


setup(
    name='space_navigator',
    version='1.0',
    description='Space Navigator - a tool, for spacecraft collision avoidance using RL.',

    url="https://github.com/yandexdataschool/satellite-collision-avoidance/",

    # Choose your license
    license='MIT',
    packages=find_packages(),
    # packages=['space_navigator'],  # same as name

    # What does your project relate to?
    keywords='machine learning, reinforcement learning, collision avoidance, spacecraft',

    # install_requires=[
    #     'pykep == 2.1',
    #     'pandas >= 0.20.3',
    #     'matplotlib >= 2.1.0',
    #     'numpy >= 1.14.2',
    #     'scipy >= 1.0.1'
    # ],
)
