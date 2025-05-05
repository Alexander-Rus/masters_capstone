from setuptools import setup, find_packages

setup(
    name="trackmania_rl",
    version="0.1.0",
    description="Simplified Trackmania reinforcement learning agent",
    author="Your Name",
    license="MIT",
    packages=find_packages(include=["trackmania_rl", "trackmania_rl.*"]),
    python_requires=">=3.10,<3.12",
    install_requires=[
        "numpy",
        "torch",
        "opencv-python",
        "matplotlib",
        "joblib",
        "pygbx",  # if map parsing is still used
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "pytest",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
