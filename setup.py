from setuptools import setup
from setuptools.depends import get_module_constant

version = get_module_constant("wavcmp", "__version__", default=None)
assert version

setup(
    name="wavcmp",
    version=version,
    description="compare audio files",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    keywords="audio compare match offset padding",
    url="https://github.com/zholos/wavcmp",
    author="Andrey Zholos",
    author_email="aaz@q-fu.com",
    license="MIT",
    packages=["wavcmp"],
    entry_points={
        "console_scripts": ["wavcmp=wavcmp.cmdline:main"]
    },
    install_requires=["numpy", "scipy"]
)
