import subprocess
import setuptools

from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        script = "../setup/download.sh"
        proc = subprocess.Popen(["bash", script], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        print(out)
        print(err)


setuptools.setup(
    name="comet-commonsense",
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
    cmdclass={'install': PostInstallCommand, 'develop': PostInstallCommand}
)
