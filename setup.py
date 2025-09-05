from setuptools import setup


setup(
    name='hustle-tools',
    version='0.0.1',
    author='Abby Boehm and Carlos Gascon',
    url='https://github.com/Exo-TiC/HUSTLE-tools',
    license='MIT',
    packages=['hustle_tools','hustle_tools.read_and_write_config','hustle_tools.plotting',
              'hustle_tools.stage_0','hustle_tools.stage_1','hustle_tools.stage_2',
              'hustle_tools.stage_3',],
    description='HST UVIS reduction pipeline',
    long_description="Pipeline for analysis of Hubble Space Telescope "
                     "WFC3-UVIS G280 spectroscopic observations.",
    python_requires='>=3.8.0',
    install_requires=['scipy>=1.8.0', 'numpy', 'xarray', 'astroquery', 'astropy',
                      'photutils<=1.13.0', 'matplotlib', 'tqdm', 'grismconf', 'wfc3tools',],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    zip_safe=True,
)
