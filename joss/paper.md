---
title: 'HUSTLE-tools: a one stop shop for Hubble WFC3-UVIS/G280 spectral reduction'
tags:
  - Python
  - astronomy
  - transiting exoplanets
authors:
  - name: Abby Boehm
    orcid: 0000-0002-4945-1860
    affiliation: '1'
    corresponding: true
  - name: Carlos Gascon
    orcid: 0000-0001-5097-9251
    affiliation: '2'
    corresponding: true
  - name: David Grant
    orcid: 0000-0001-5878-618X
    affiliation: '3'
    corresponding: false
  - name: Hannah R Wakeford
    orcid: 0000-0003-4328-3867
    affiliation: '3'
    corresponding: false
  - name: Nikole K Lewis
    orcid: 0000-0002-8507-1304
    affiliation: '1'
    corresponding: false
affiliations:
  - name: Department of Astronomy and Carl Sagan Institute, Cornell University, 122 Sciences Drive, Ithaca, NY, 14853, USA
    index: 1
  - name: Center for Astrophysics | Harvard & Smithsonian, 60 Garden Street, Cambridge MA 02138, USA
    index: 2
  - name: HH Wills Physics Laboratory, University of Bristol, Tyndall Avenue, Bristol, BS8 1TL, UK
    index: 3
date: 16 June 2025
bibliography: paper.bib

---


# Summary

Fully understanding the complex physical and chemical processes that shape exoplanet atmospheres requires the complete spectrum from the ultraviolet (UV) to the mid-infrared (IR). In particular, the UV-optical has proven vital to constraining the presence of UV absorbers or scatterers [e.g. @lothringer2022], atmospheric escape [e.g. @bourrier2018], and enhanced scattering due to aerosol opacities [@ohno2020].

The Hubble Space Telescope Wide Field Camera 3 UV Imaging Spectrograph (HST WFC3-UVIS) G280 grism is the only low-resolution spectrograph that gives us access to UV-optical wavelengths (0.2--0.8 microns) simultaneously. With most current instrumentation operating in near- and mid-IR wavelengths, and UV successors to HST still many years away, HST WFC3-UVIS/G280 emerges as the only instrument capable of unveiling the physics and chemistry that takes place in the upper atmospheric layers of exoplanets. Developed in the context of the Hubble Ultraviolet-optical Survey of Transiting Legacy Exoplanets (HUSTLE, GO-17183 PI: Wakeford), HUSTLE-tools is an open-source package of modules designed to easily download, organize, analyze, clean, and extract the target spectrum from HST WFC3-UVIS/G280 spectral images.


# Statement of Need

While the Hubble Space Telescope (HST) has been operating for more than 30 years, the WFC3-UVIS/G280 mode saw limited use until the recent surge of application to transmission spectroscopy of exoplanets [@wakeford2020; @lewis2020]. Current HST WFC3 pipelines [e.g. @eureka; @pacman] only service WFC-IR spectroscopy. HST WFC3-UVIS/G280 observations offer unique challenges in extracting spectral information: a curved spectral trace with a varied width, overlapping spectral orders, and high cosmic ray counts. These are in addition to potential variations in spectral extraction based on the use of different sub-array sizes and positions on the detector for each observation. Such challenges make current HST pipelines not suitable for reducing UVIS/G280 observations, and therefore a specialized pipeline is needed. 


# Design and Features

Similar to other HST and JWST pipelines [@eureka; @exotedrf], HUSTLE-tools is built in a modular fashion consisting of three stages. These stages encompass the different steps and subroutines necessary for data quality assessment, reduction, and spectral extraction:

* Stage 0: allows the user to download and organize the files included as part of a specific GTO program number and visit number within that program. This stage can produce a "quick look" .gif file displaying all the downloaded G280 frames, intended to be used to diagnose potential errors during the observation. 

* Stage 1: allows the user to perform a suite of cleaning operations on the data to treat cosmic rays, hot pixels, and background signal, as well as track the motion of the trace across the detector between frames. The output of this stage contains the cleaned frames together with several auxiliary variables such as the image displacement in X and Y detector axes.

* Stage 2: allows the user to extract spectra from each image. The data are calibrated using the GRISMCONF package [@pirzkal2020], which offers wavelength solutions and trace positions for orders up to +/-4. Spectral extraction can be performed using a standard unweighted aperture of uniform size, or via the optimal weighting method described in @horne1986 and @marsch1989. The output of this stage is the extracted 1D spectral timeseries.

Each run for each stage is defined through a ".hustle" configuration file, where all the parameters and variables needed to perform the stage subroutines are defined. For reproducibility, a copy of each configuration file is saved within the run's output directory. These files ensure that the user can always recreate a past result and facilitate comparing different runs.

Included tutorials and example scripts ensure HUSTLE-tools can be run with minimal prior experience in pipeline development and operation. The modular and user-friendly design of HUSTLE-tools permits users to fine-tune their reduction to obtain optimal results. Users can toggle and tweak their desired processes within each stage and can easily rerun stages to explore different reduction techniques.

HUSTLE-tools is built from the 'Hazelnut' pipeline presented in @Boehm2024 and the 'lluvia' pipeline presented in @Gascon2025. 

# Acknowledgments
We acknowledge contributions from the full HUSTLE team, and specifically testing by Ailsa Campbell. 
V.A.B was funded through program number HST-GO-17183 provided through a grant from the STScI under NASA contract NAS5-26555.
C.G. was funded by La Caixa Fellowship and the Agency for Management of University and Research Grants from the Government of Catalonia (FI AGAUR).
D.G. and H.R.W were funded by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee as part of an ERC Starter Grant [grant number EP/Y006313/1].
This work benefited from the 2024 Exoplanet Summer Program in the Other Worlds Laboratory (OWL) at the University of California, Santa Cruz, a program funded by the Heising-Simons Foundation and NASA.

# References
