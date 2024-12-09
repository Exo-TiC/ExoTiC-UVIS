from setuptools import setup


setup(
    name='exotic-uvis',
    version='0.0.1',
    author='Abby Boehm and Carlos Gascon',
    url='https://github.com/Exo-TiC/ExoTiC-UVIS',
    license='MIT',
    packages=['exotic_uvis','exotic_uvis.read_and_write_config','exotic_uvis.plotting',
              'exotic_uvis.stage_0','exotic_uvis.stage_1','exotic_uvis.stage_2',
              'exotic_uvis.stage_3',],
    description='HST UVIS reduction pipeline',
    long_description="Pipeline for analysis of Hubble Space Telescope "
                     "WFC3-UVIS G280 spectroscopic observations.",
    python_requires='>=3.8.0',
    install_requires=['scipy>=1.8.0', 'numpy', 'xarray', 'astroquery', 'astropy',
                      'photutils', 'matplotlib', 'tqdm', 'grismconf', 'wfc3tools','jwst',],
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
