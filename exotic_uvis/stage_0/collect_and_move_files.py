import os
import shutil
import glob

from astropy.io import fits

def move_files(fromdir, outdir):
    '''
    Moves the spec, direct, and misc files to the right directories.

    :param fromdir: str. The directory where the orbitNframeN, orbitNdirectN, and misc files are kept.
    :param outdir: str. Where the files will be moved to.
    :return: files removed from their current path and sent to the target_dir. No callables.
    '''
    # Create the output directory.
    if not os.path.exists(outdir):
        print("Creating output directory {}...".format(outdir))
        os.makedirs(outdir)

    # Sort files by spec, direct, and misc.
    files = sorted(glob.glob(os.path.join(fromdir, "*")))
    spec_files = [f for f in files if ("frame" in f and ".fits" in f)]
    direct_files = [f for f in files if "direct" in f]
    misc_files = [f for f in files if (f not in spec_files and f not in direct_files)]

    for files, target in zip((spec_files, direct_files, misc_files),("specimages","directimages","miscfiles")):
        targetdir = os.path.join(outdir,target)
        if not os.path.exists(targetdir):
            os.makedirs(targetdir)
        print("Moving {} listed files to target directory {}...".format(len(files),targetdir))
    
        for i, f in enumerate(files):
            split_filename = str.split(f, sep="/")
            shutil.move(f, os.path.join(targetdir, split_filename[-1]))
        print("All files listed moved into {}.".format(targetdir))
    print("All spec, direct, and misc files moved.")

def identify_orbits(spec_flt, spec_spt, direct_flt, direct_spt, misc_files):
    '''
    Opens each file and checks exposure time starts to find orbits.

    :param spec_flt: lst of str. The filepaths to the spectroscopic flt images. Used to find orbits.
    :param spec_spt: lst of str. The filepaths to the spectroscopic spt images corresponding to the flt images.
    :param direct_flt: lst of str. The filepaths to the direct flt images.
    :param direct_spt: lst of str. The filepaths to the direct spt images corresponding to the flt images.
    :param misc_files: lst of str. The filepaths to the miscellanous files.
    :return: spec, direct, and misc files all renamed to have orbit#frame# tags.
    '''
    # First, sort all files by exposure time and get corresponding file prefix names.
    starts = []
    prefixes = []
    for f in spec_flt:
        filename = str.split(f, sep="/")
        split_filename = str.split(filename[-1], sep='_')
        with fits.open(f) as fits_file:
            starts.append(fits_file[0].header["EXPSTART"]*86400) # turn it into seconds
            prefixes.append(split_filename[0]) # this is the iexr##xxxx part of the filename, which can be used to find associated files
    bundle = [(i,j,) for i,j, in zip(starts,prefixes)]
    bundle = sorted(bundle, key = lambda x: x[0]) # sorted by exposure time

    # Create association between iexr##xxxx and orbit#frame#.
    rename = {bundle[0][1]:"orbit1frame1"}

    # Now detect jumps in exposure start time, which are expected to be separated by >45 minutes.
    orbit_N = 1
    frame_N = 1
    for i in range(1,len(bundle)):
        exp_jump = bundle[i][0] - bundle[i-1][0]
        if exp_jump > 45*60: # if greater than 45 minutes
            # Increase orbit number and reset frame number.
            orbit_N += 1
            frame_N = 1
        else:
            # We are still in the same orbit, so increment the frame number.
            frame_N += 1
        rename[bundle[i][1]] = "orbit{}frame{}".format(orbit_N,frame_N)
    print("Detected %.0f orbits and created new filenames to update." % orbit_N)

    # Now we need to replace instances of iexr##xxxx in filenames with orbit#frame#.
    print("Renaming spec and misc files.")
    for prefix in prefixes:
        for files in (spec_flt, spec_spt, misc_files):
            relevant_files = [f for f in files if prefix in f]
            for f in relevant_files:
                f_new = str.replace(f, prefix, rename[prefix])
                shutil.move(f, f_new)
    
    # Direct images do not follow this convention. So we do it all again.
    starts = []
    prefixes = []
    for f in direct_flt:
        filename = str.split(f, sep="/")
        split_filename = str.split(filename[-1], sep='_')
        with fits.open(f) as fits_file:
            starts.append(fits_file[0].header["EXPSTART"]*86400) # turn it into seconds
            prefixes.append(split_filename[0]) # this is the iexr##xxxx part of the filename, which can be used to find associated files
    bundle = [(i,j,) for i,j, in zip(starts,prefixes)]
    bundle = sorted(bundle, key = lambda x: x[0]) # sorted by exposure time

    # Create association between iexr##xxxx and orbit#frame#.
    rename = {bundle[0][1]:"orbit1direct1"}

    if len(direct_flt) == 1:
        # There is only one direct image, so no more work required.
        pass
    else:
        # Now detect jumps in exposure start time, which are expected to be separated by >45 minutes.
        orbit_N = 1
        frame_N = 1
        for i in range(1,len(bundle)):
            exp_jump = bundle[i][0] - bundle[i-1][0]
            if exp_jump > 45*60: # if greater than 45 minutes
                # Increase orbit number and reset frame number.
                orbit_N += 1
                frame_N = 1
            else:
                # We are still in the same orbit, so increment the direct frame number.
                frame_N += 1
            rename[bundle[i][1]] = "orbit{}direct{}".format(orbit_N,frame_N)
    print("Detected %.0f orbits and created new filenames to update." % orbit_N)

    # Now we need to replace instances of iexr##xxxx in filenames with orbit#direct#.
    print("Renaming spec and misc files.")
    for prefix in prefixes:
        for files in (direct_flt, direct_spt, misc_files):
            relevant_files = [f for f in files if prefix in f]
            for f in relevant_files:
                f_new = str.replace(f, prefix, rename[prefix])
                shutil.move(f, f_new)
    
    print("Renamed all files to follow orbit#frame# or direct#frame# convention.")

def collect_files(search_dir):
    '''
    Searches the search_dir for files.

    :param search_dir: str. The directory that you want to locate files in.
    :return: lists of filepaths inside of the directory sorted by direct/spec/misc.
    '''
    print("Collecting and categorizing files from {}...".format(search_dir))
    
    # Initialize lists
    # Spectroscopic image files using the G280, split by flt and spt
    spec_flt = []
    spec_spt = []
    # Direct image files using F filters, split by flt and spt
    direct_flt = []
    direct_spt = []
    # Files that did not fit into any other category
    misc_files = []

    # Glob files
    files = sorted(glob.glob(os.path.join(search_dir, "*")))
    
    # Sort files into direct, spec, and misc.
    for f in files:
        if "fits" not in f:
            # Immediately identify as misc file
            misc_files.append(f)
            continue
        with fits.open(f) as fits_file:
            if "spt.fits" in f:
                filter_type = fits_file[0].header["SS_FILT"]
                if filter_type == "G280":
                    # Spec type
                    spec_spt.append(f)
                elif "F" in filter_type:
                    # Direct type
                    direct_spt.append(f)
                else:
                    # Unrecognized filter
                    misc_files.append(f)
            elif "flt.fits" in f:
                filter_type = fits_file[0].header["FILTER"]
                if filter_type == "G280":
                    # Spec type
                    spec_flt.append(f)
                elif "F" in filter_type:
                    # Direct type
                    direct_flt.append(f)
                    # Direct type
                else:
                    # Unrecognized filter
                    misc_files.append(f)
            else:
                # Unrecognizd file type
                misc_files.append(f)

    print("Collected spec, direct, and misc files.")
    return spec_flt, spec_spt, direct_flt, direct_spt, misc_files