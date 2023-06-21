from setuptools import find_packages, setup

setup(
    name='rl_test',
    version='0.1',
    description='rl test packiage',
    author='Tobby Lie',
    author_email='tobbylie@gmail.com',
    packages=find_packages(),
    install_requires=[
        "click",
        "gym",
        "matplotlib",
        "numpy",
        "opencv-python",
        "pillow",
        "ptan",
        "pygame",
        "scipy",
        "torch",
        "torchvision",
        "tensorboard-pytorch",
        "tensorflow",
        "tensorboard",
        "tqdm",
    ],
)
