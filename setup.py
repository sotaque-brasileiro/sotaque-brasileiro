from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sotaque_brasileiro",
    version="0.1.11",
    license="GPL-3.0",
    description="Sotaque Brasileiro é uma base de dados para estudo de regionalismos brasileiros através da voz.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author="Gabriel Gazola Milan",
    author_email="gabriel.gazola@poli.ufrj.br",
    url="https://github.com/gabriel-milan/sotaque-brasileiro",
    install_requires=[
        "scipy",
        "pandas",
        "numpy",
        "plotly",
        "google-cloud-storage==1.42.1",
        "requests==2.21.0",
        "SpeechRecognition==3.8.1",
        "pydub==0.25.1",
        "webrtcvad==2.0.10",
        "librosa==0.8.1",
        "python_speech_features==0.6",
        "p-tqdm==1.3.3",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
