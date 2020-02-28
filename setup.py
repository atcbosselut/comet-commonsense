import os
import subprocess
import setuptools

from setuptools.command.install import install
from setuptools.command.develop import develop


# Are we building from the repository or from a source distribution?
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'comet')
BUILD_DIR = SRC_DIR if os.path.exists(SRC_DIR) else os.path.join(ROOT_DIR, '../..')


with open("README.md", "r") as fh:
    long_description = fh.read()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        self.execute(_post_install, (), msg="Running post install task")


class PostDevelopCommand(develop):
    """Post-installation for installation mode."""
    def run(self):
        develop.run(self)
        self.execute(_post_install, (), msg="Running post install task")


def _post_install():
    script = os.path.join(BUILD_DIR, os.pardir, "setup", "download.sh")
    subprocess.call(["chmod", "+x", script])
    subprocess.call(["/bin/bash", "-c", script])


setuptools.setup(
    name="comet2",
    version="2.0",
    author="This version by Vered Shwartz. Original version by Antoine Bosselut.",
    description="COMET: Commonsense Transformers for Automatic Knowledge Graph Construction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vered1986/comet-commonsense",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "torchvision",
        "transformers==2.4.1",
        "pandas",
        "ftfy",
        "spacy",
        "tensorboardX",
        "nltk",
        "gdown"
    ],
    python_requires='>=3.6',
    cmdclass={'install': PostInstallCommand, 'develop': PostDevelopCommand}
)
