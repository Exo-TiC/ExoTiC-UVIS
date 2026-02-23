import os
import shutil
import glob

from astropy.io import fits


def collect_and_move_files(visit_number, fromdir, outdir, verbose=2):
    """Collects, renames, and moves the spec, direct, visit-related, and misc files to the right directories.

    Args:
        visit_number (str): The visit number that we want to look at.
        fromdir (str): The directory where the orbitNframeN, orbitNdirectN, and misc files are kept.
        outdir (str): Where the files will be moved to.
        verbose (int, optional): From 0 to 2, how much detail you want the output logs to have. Defaults to 2.
    """
    # Create the output directory.
    if not os.path.exists(outdir):
        if verbose == 1:
            print("Creating output directory {}...".format(outdir))
        os.makedirs(outdir)
    
    # Collect and sort files from the fromdir that are in the right visit.
    spec_flt, spec_spt, direct_flt, direct_spt, jit_files, misc_files = collect_files(fromdir, visit_number, verbose)

    # Then sort by orbit.
    identify_orbits(spec_flt, spec_spt, direct_flt, direct_spt, jit_files, misc_files, outdir, verbose)
    
    # Now re-sort using the updated filenames.
    files = sorted(glob.glob(os.path.join(fromdir, "*")))

    # First, catch spec files (marked with fm) and only take spt and flt file.
    spec_files = [f for f in files if ("fm" in f and ".fits" in f and "hst_" not in f)]
    spec_files = [f for f in spec_files if ("flt.fits" in f or "spt.fits" in f)]
    # Then, catch direct files (marked with dr) and only take spt and flt files.
    direct_files = [f for f in files if ("dr" in f and ".fits" in f and "hst_" not in f)]
    direct_files = [f for f in direct_files if ("flt.fits" in f or "spt.fits" in f)]
    # Catch all jitter files.
    jit_files = [f for f in files if ("jit" in f and ".fits" in f and "hst_" not in f)]
    # Now catch files that were at least identified with an orbit of interest.
    visit_files = [f for f in files if ("or" in f and "hst_" not in f)]
    visit_files = [f for f in visit_files if ("fm" in f or "dr" in f)]
    visit_files = [f for f in visit_files if (f not in spec_files and f not in direct_files and f not in jit_files)]
    # Finally, filter everything else.
    misc_files = [f for f in files if (f not in spec_files and f not in direct_files and f not in jit_files and f not in visit_files)]
    # Remove directories!
    misc_files = [f for f in misc_files if not os.path.isdir(f)]

    for files, target in zip((spec_files, direct_files, jit_files, visit_files, misc_files),
                             ("specimages","directimages","jitterfiles","visitfiles","miscfiles")):

        targetdir = os.path.join(outdir,target)
        if not os.path.exists(targetdir):
            os.makedirs(targetdir)
        if verbose == 2:
            print("Moving {} listed files to target directory {}...".format(len(files),targetdir))
    
        for i, f in enumerate(files):
            split_filename = str.split(f, sep="/")
            shutil.move(f, os.path.join(targetdir, split_filename[-1]))
        if verbose == 2:
            print("All files listed moved into {}.".format(targetdir))
    if verbose >= 1:
        print("All spec, direct, jitter, and misc files moved.")


def identify_orbits(spec_flt, spec_spt, direct_flt, direct_spt, jit_files, misc_files, outdir, verbose=2):
    """Opens each file and checks exposure time starts to find orbits.

    Args:
        spec_flt (lst of str): The filepaths to the spectroscopic flt images. Used to find orbits.
        spec_spt (lst of str): The filepaths to the spectroscopic spt images corresponding to the flt images.
        direct_flt (lst of str): The filepaths to the direct flt images.
        direct_spt (lst of str): The filepaths to the direct spt images corresponding to the flt images.
        jit_files (lst of str): The filepaths to the jitter files, one for each orbit.
        misc_files (lst of str): The filepaths to the miscellanous files.
        outdir (str): Where the files will be moved to, or where the .flt files already are.
        verbose (int, optional): From 0 to 2, how much detail you want the output logs to have. Defaults to 2.
    """
    # First, sort all files by exposure time and get corresponding file prefix names.
    starts = []
    prefixes = []
    for f in spec_flt:
        with fits.open(f) as fits_file:
            starts.append(fits_file[0].header["EXPSTART"]*86400) # turn it into seconds
            filename = fits_file[0].header["FILENAME"]
            split_filename = str.split(filename, sep='_')
            prefixes.append(split_filename[0]) # this is the iexr##xxx part of the filename, which can be used to find associated files
    bundle = [(i,j,) for i,j, in zip(starts,prefixes)]
    bundle = sorted(bundle, key = lambda x: x[0]) # sorted by exposure time

    if spec_flt == []:
        # If the files were already downloaded, we need to go fetch them and create the associations with iexr##xxx
        if verbose >= 1:
            print("Spectroscopic images were already downloaded: reading off FILENAME keyword to create iexr##xxx filename associations...")
        specimages_path = os.path.join(outdir,"specimages")
        spec_flt = sorted(glob.glob(os.path.join(specimages_path,"*_flt.fits")))
        if spec_flt == []:
            # Throw an error: we need to have *flt.fits files before we can parse any other file in the program
            raise FileNotFoundError("*flt.fits files not located; please download these first as HUSTLE-Tools requires them to parse other files for your program ID + target!")
        for f in spec_flt:
            with fits.open(f) as fits_file:
                starts.append(fits_file[0].header["EXPSTART"]*86400) # turn it into seconds
                filename = fits_file[0].header["FILENAME"]
                split_filename = str.split(filename, sep='_')
                prefixes.append(split_filename[0]) # this is the iexr##xxx part of the filename, which can be used to find associated files
        bundle = [(i,j,) for i,j, in zip(starts,prefixes)]
        bundle = sorted(bundle, key = lambda x: x[0]) # sorted by exposure time

    # Create association between iexr##xxx and or##fm###.

    rename = {bundle[0][1]:"or01fm001"}

    # Now detect jumps in exposure start time, which are expected to be separated by >35 minutes.
    orbit_N = 1
    frame_N = 1
    for i in range(1,len(bundle)):
        exp_jump = bundle[i][0] - bundle[i-1][0]
        if exp_jump > 35*60: # if greater than 35 minutes
            # Increase orbit number and reset frame number.
            orbit_N += 1
            frame_N = 1
        else:
            # We are still in the same orbit, so increment the frame number.
            frame_N += 1
        o_type = str(orbit_N).zfill(2)
        f_type = str(frame_N).zfill(3)
        rename[bundle[i][1]] = "or{}fm{}".format(o_type,f_type)
    
    if verbose == 2:
        print("Detected %.0f orbits and created new filenames to update." % orbit_N)

    # Now we need to replace instances of iexr##xxxx in filenames with or##fm###.
    if verbose == 2:
        print("Renaming spec/misc files.")
    for prefix in prefixes:
        for files in (spec_flt, spec_spt, misc_files):
            relevant_files = [f for f in files if prefix in f]
            for f in relevant_files:
                f_new = str.replace(f, prefix, rename[prefix])
                if not os.path.exists(f_new):
                    shutil.move(f, f_new)
                else:
                    if verbose == 2:
                        print(f"{f_new} has already been downloaded; removing excess copy of file.")
                    os.rmdir(f)
    
    # Direct images do not follow this convention. So we do it all again.
    starts = []
    prefixes = []
    for f in direct_flt:
        with fits.open(f) as fits_file:
            starts.append(fits_file[0].header["EXPSTART"]*86400) # turn it into seconds
            filename = fits_file[0].header["FILENAME"]
            split_filename = str.split(filename, sep='_')
            prefixes.append(split_filename[0]) # this is the iexr##xxx part of the filename, which can be used to find associated files
    bundle = [(i,j,) for i,j, in zip(starts,prefixes)]
    bundle = sorted(bundle, key = lambda x: x[0]) # sorted by exposure time

    if direct_flt == []:
        # If the files were already downloaded, we need to go fetch them and create the associations with iexr##xxx
        if verbose >= 1:
            print("Direct images were already downloaded: reading off FILENAME keyword to create iexr##xxx filename associations...")
        directimages_path = os.path.join(outdir,"directimages")
        direct_flt = sorted(glob.glob(os.path.join(directimages_path,"*_flt.fits")))
        if direct_flt == []:
            # Throw an error: we need to have *flt.fits files before we can parse any other file in the program
            raise FileNotFoundError("*flt.fits files not located; please download these first as HUSTLE-Tools requires them to parse other files for your program ID + target!")
        for f in direct_flt:
            with fits.open(f) as fits_file:
                starts.append(fits_file[0].header["EXPSTART"]*86400) # turn it into seconds
                filename = fits_file[0].header["FILENAME"]
                split_filename = str.split(filename, sep='_')
                prefixes.append(split_filename[0]) # this is the iexr##xxx part of the filename, which can be used to find associated files
        bundle = [(i,j,) for i,j, in zip(starts,prefixes)]
        bundle = sorted(bundle, key = lambda x: x[0]) # sorted by exposure time

    # Create association between iexr##xxx and or##dr###.
    rename = {bundle[0][1]:"or01dr001"}

    if len(direct_flt) == 1:
        # There is only one direct image, so no more work required.
        pass
    else:
        # Now detect jumps in exposure start time, which are expected to be separated by >35 minutes.
        orbit_N = 1
        frame_N = 1
        for i in range(1,len(bundle)):
            exp_jump = bundle[i][0] - bundle[i-1][0]
            if exp_jump > 35*60: # if greater than 35 minutes
                # Increase orbit number and reset frame number.
                orbit_N += 1
                frame_N = 1
            else:
                # We are still in the same orbit, so increment the direct frame number.
                frame_N += 1
            o_type = str(orbit_N).zfill(2)
            f_type = str(frame_N).zfill(3)
            rename[bundle[i][1]] = "or{}dr{}".format(o_type,f_type)
    if verbose == 2:
        print("Detected %.0f orbits and created new filenames to update." % orbit_N)
    # Jitter files follow the convention of iexr##NNN where NNN is a number, not a string of letters.
    if verbose == 2:
        print("Renaming jitter files.")
    for i, jitter in enumerate(list(sorted(jit_files))):
        # Get the jitter file's current name.
        f = str.split(jitter, sep="/")
        f_new = [i for i in f]

        # Rename it based on orbit number. There will be one per orbit!
        o_type = str(i+1).zfill(2)
        new_jitter_name = "or{}jt000_jit.fits".format(o_type)
        if i == len(sorted(jit_files))-1:
            # This is the direct image jitter.
            new_jitter_name = "or01dr000_jit.fits"

        # Replace the jitter name and move.
        f_new[-1] = new_jitter_name
        shutil.move(os.path.join(f[0],f[1]), os.path.join(f_new[0],f_new[1]))
    if verbose == 2:
        print("Relocated %.0f jitter files and created new filenames to update." % len(jit_files))

    # Now we need to replace instances of iexr##xxx in filenames with or##dr###.
    if verbose == 2:
        print("Renaming direct/misc files.")
    for prefix in prefixes:
        for files in (direct_flt, direct_spt, misc_files):
            relevant_files = [f for f in files if prefix in f]
            for f in relevant_files:
                f_new = str.replace(f, prefix, rename[prefix])
                if not os.path.exists(f_new):
                    shutil.move(f, f_new)
                else:
                    if verbose == 2:
                        print(f"{f_new} has already been downloaded; removing excess copy of file.")
                    os.rmdir(f)
    
    if verbose >= 1:
        print("Renamed all files to follow or##fm### (for spec frames) or or##dr### (for direct frames) convention.")


def collect_files(search_dir, visit_number, verbose=2):
    """Searches the search_dir for files of the right visit_number.

    Args:
        search_dir (str): The visit number that we want to look at.
        visit_number (str): The directory that you want to locate files in.
        verbose (int, optional): From 0 to 2, how much detail you want the output logs to have. Defaults to 2.

    Returns:
        lst of str: Lists of filepaths inside of the directory sorted by direct/spec/misc.
    """
    # Start the routine
    if verbose >= 1:
        print("Collecting and categorizing files from {}...".format(search_dir))

    # Define some filename info that is an immediate reject
    reject = ("f_flt","f_spt","hst_")
    # And some info that must be present to retain
    require = (".fits",visit_number)
    
    # Initialize lists
    # Spectroscopic image files using the G280, split by flt and spt
    spec_flt = []
    spec_spt = []
    # Direct image files using F filters, split by flt and spt
    direct_flt = []
    direct_spt = []

    # Jitter files, one per orbit
    jit_files = []

    # Files that did not fit into any other category
    misc_files = []

    # Glob files
    files = sorted(glob.glob(os.path.join(search_dir, "*")))
  
    # Sort files into direct, spec, jitter, and misc

    for f in files:
        if os.path.isdir(f):
            if verbose == 2:
                print(f"{f} is a directory; shutil will not move it.")
            continue
        if any(txt in f for txt in reject):
            # It's a file we do not want.
            misc_files.append(f)
            continue
        if any(txt not in f for txt in require):
            # It's not one of the files we definitely want.
            misc_files.append(f)
            continue
        # Otherwise, we can look at it a little closer.
        with fits.open(f) as fits_file:
            if "jit.fits" in f:
                # jit files contain the jitter information for decorrelation
                jit_files.append(f)
                continue
            # Then we must check if it has the detector key

            try:
                if fits_file[0].header["DETECTOR"] != "UVIS":
                    # Wrong detector so put it in misc
                    misc_files.append(f)
                    continue
            except KeyError:
                # It has no detector at all so put it in misc
                misc_files.append(f)
                continue
            if "spt.fits" in f:
                # spt files are used for "embedding" and jitter decorrelation.
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
                # flt files contain the calwfc3-corrected images from MAST.
                filter_type = fits_file[0].header["FILTER"]
                if filter_type == "G280":
                    # Spec type
                    spec_flt.append(f)
                elif "F" in filter_type:
                    # Direct type
                    direct_flt.append(f)
                else:
                    # Unrecognized filter
                    misc_files.append(f)
            else:
                # Unrecognizd file type
                misc_files.append(f)
    if verbose == 1:
        print("Collected spec, direct, jitter, and misc files.")
    if verbose == 2:
        print("Collected {} spec_flt files.".format(len(spec_flt)))
        print("Collected {} spec_spt files.".format(len(spec_spt)))
        print("Collected {} direct_flt files.".format(len(direct_flt)))
        print("Collected {} direct_spt files.".format(len(direct_spt)))
        print("Collected {} jit_files files.".format(len(jit_files)))
        print("Collected {} misc_files files.".format(len(misc_files)))
    return spec_flt, spec_spt, direct_flt, direct_spt, jit_files, misc_files
