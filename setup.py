import setuptools


setuptools.setup(
    name="slhfdtd",
    author="Saleh Shamloo Ahmadi",
    author_email="slhshamloo@gmail.com",
    url="https://github.com/slhshamloo/maxwell-fdtd",
    packages=setuptools.find_packages(),
    install_requires= ["numpy", "matplotlib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ]
)
