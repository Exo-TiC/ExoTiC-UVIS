from wfc3tools import sub2full, embedsub
import os

from astropy.io import fits

def embed(files):
    '''
    Embeds the provided *_flt.fits files using the *_spt.fits files that live
    in the same directory as the *_flt.fits files. Also adds a new [0].header keyword
    to each fits file that reports the subarray location.
    
    :param files: lst of str. The paths to each of the files you want to embed.
    :return: lst of str of all the paths to the embedded files. In the same dir as where the *_flt.fits and *_spt.fits files were stored, embedded *f_flt.fits files that have been embedded in the full frame and have had subarray coords written into the headers.

    !!NEW!! Usse sub2full to determine the subarray offset, needed to properly configure
    the wavelength solution.
    :param files: lst of str. The paths to each of the files you want to attach subarray offset coordinates to.
    :return: all *_flt.fits files with new header info added into "SUBARRN#" keywords, where N = [X,Y] and # = [1,2].
    '''
    print("Writing subarray coordinate offsets into *_flt.fits files...")
    for f in files:
        # Need to grab the subarray extent in the full frame.
        y1,y2,x1,x2 = sub2full(f, fullExtent=True)[0]

        # Open the file and write this info to its header.
        with fits.open(f) as fits_file:
            fits_file[0].header["SUBARRX1"] = (x1, 'Subarray coordinate x1 (left edge)')
            fits_file[0].header["SUBARRX2"] = (x2, 'Subarray coordinate x2 (right edge)')
            fits_file[0].header["SUBARRY1"] = (y1, 'Subarray coordinate y1 (bottom edge)')
            fits_file[0].header["SUBARRY2"] = (y2, 'Subarray coordinate y2 (top edge)')
            fits_file.writeto(f, overwrite=True)
    print("Wrote offset values to %.0f files." % len(files))

'''
   outfiles = []  
   for f in files:
        # Need to grab the subarray extent in the full frame.
        y1,y2,x1,x2 = sub2full(f, fullExtent=True)[0]   

        # Run the _flt.fits files through embedsub to place them in the full image using the _spt.fits files.
        embedsub(f)
        
        # Open the embedded f_flt.fits file that was just created and add a new header to it.
        f_emb = os.path.join(os.path.dirname(f), os.path.basename(f)[0:8] + 'f_flt.fits')
        with fits.open(f_emb) as fits_file:
            fits_file[0].header["SUBARRX1"] = (x1, 'Subarray coordinate x1 (left edge)')
            fits_file[0].header["SUBARRX2"] = (x2, 'Subarray coordinate x2 (right edge)')
            fits_file[0].header["SUBARRY1"] = (y1, 'Subarray coordinate y1 (bottom edge)')
            fits_file[0].header["SUBARRY2"] = (y2, 'Subarray coordinate y2 (top edge)')
            fits_file.writeto(f_emb, overwrite=True)
        outfiles.append(f_emb)
    return outfiles
'''