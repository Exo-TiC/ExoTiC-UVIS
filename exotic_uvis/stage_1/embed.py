from wfc3tools import sub2full

def embed(files):
    '''
    Use sub2full to determine the subarray offset, needed to properly configure the wavelength solution.

    :param files: lst of str. The paths to each of the files you want to attach subarray offset coordinates to.
    :return: lst of lsts of ints, each lst being the subarray coordinates for the corresponding frame in files.
    '''
    print("Reading subarray coordinate offsets with sub2full and *_flt.fits files...")
    subarr_coords = []
    for f in files:
        # Need to grab the subarray extent in the full frame.
        y1,y2,x1,x2 = sub2full(f, fullExtent=True)[0]
        # And append.
        subarr_coords.append([y1,y2,x1,x2])
    print("Read offset values from %.0f files." % len(subarr_coords))
    return subarr_coords