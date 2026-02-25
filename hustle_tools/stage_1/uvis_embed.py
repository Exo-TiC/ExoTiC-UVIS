def get_subarr_coords(hdul):
    """Uses SCI header keywords to get subarray coordinate information independent of spt files and the wfc3tools module. Adapted from https://github.com/spacetelescope/hst_notebooks/blob/main/notebooks/WFC3/uvis_g280_transit/g280_transit_tools.py which contains the function embedsub_uvis.

    Args:
        hdul (fits file): opened *_flt.fits G280 fits file.

    Returns:
        lst: list of all of the subarr coordinates.
    """
    subarr_coords = []

    # fetch relevant keywords
    for i in (1,2):
        naxis = hdul['SCI'].header['NAXIS{}'.format(i)]
        ltv = hdul['SCI'].header['LTV{}'.format(i)]

        x1,x2 = int(-ltv), int(-ltv) + naxis
        subarr_coords.append(x1+1) # the +1 brings it into agreement with wfc3tools embedsub
        subarr_coords.append(x2)

    # and return
    return subarr_coords
