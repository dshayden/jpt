# """
# The build/compilations setup
#
# >> pip install -r requirements.txt
# >> python setup.py install
# """
import pip
import logging
import pkg_resources
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setup(
    name='jpt_tracker',
    version='0.1',
    author='David S Hayden',
    author_email='dshayden@mit.edu',
    license='MIT',
    description='Joint Posterior Multi-Object Tracker',
    packages=["jpt"],
    scripts=[
      'scripts/jpt_runPointTracker',
      'scripts/jpt_evalX_conditional',
      'scripts/jpt_samples2matlab'
    ],
    install_requires=install_reqs,
    dependency_links=[
    ],
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords="tracking computer vision bayesian",
)
