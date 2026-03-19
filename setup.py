from setuptools import setup, find_packages

setup(
    name="cleansight",
    version="0.1.0",
    description="CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models",
    author="Zhifang Zhang, Bojun Yang, Shuo He, Weitong Chen, Wei Emma Zhang, Olaf Maennel, Lei Feng, Miao Xu",
    url="https://github.com/cleansight-defense/cleansight",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "eval": ["Pillow", "tqdm"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
