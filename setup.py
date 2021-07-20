from setuptools import setup, find_packages

setup(
    name='nested_sampling',
    version='1.0.0',
    description='A python implementation of nested sampling to compute a N-dim integral of a gaussian function',
    url='https://github.com/DanieleMDiNosse/Nested_Sampling.git',
    author='Di Nosse Daniele Maria',
    author_email='danielemdinosse@gmail.com',
    license='gnu general public license',
    packages = ['nested_sampling'],
    install_requires=['numpy','tqdm', 'pandas', 'matplotlib'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
