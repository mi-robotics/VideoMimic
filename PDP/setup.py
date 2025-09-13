from setuptools import setup


setup(
    name='pdp',
    version='0.1',
    packages=['pdp'],
    install_requires=[
        'diffusers==0.32.1',
        'dill==0.3.9',
        'einops==0.8.0',
        'gym==0.25.2',
        'hydra-core==1.2.0',
        'imageio==2.35.1',
        'imageio-ffmpeg==0.5.1',
        'mujoco==3.2.3',
        'numba==0.58.1',
        'numcodecs==0.12.1',
        'numpy==1.23.3',
        'omegaconf==2.3.0',
        'pyopengl==3.1.7',
        'scipy==1.10.1',
        'tokenizers==0.20.3',
        'tqdm==4.67.1',
        'transformers==4.46.3',
        'wandb==0.19.1',
        'zarr==2.12.0',
    ],
)