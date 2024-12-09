import os

from astroquery.mast import Observations as Obs


def get_files_from_mast(programID, target_name, visit_number, outdir, token=None, extensions=None, verbose=2):
    """Queries MAST database and downloads specified files from specified
    program, target, and visit number.

    Args:
        programID (str): ID of the observing program you want to query data
        from. On MAST, referred to as "proposal_ID".
        target_name (str): Name of the target object you want to query data
        from. On MAST, referred to as "target_name".
        visit_number (str): The visit number you want to download, e.g. "01", "02", etc.
        outdir (str): The directory you want the files downloaded to.
        token (str, optional): A MAST authentication token, if you are downloading
        proprietary data. Defaults to None.
        extensions (lst of str, optional): File extensions you want to download.
        If None, take all file extensions. Otherwise, take only the files specified.
        Defaults to None.
        verbose (int, optional): From 0 to 2, how much detail you want the
        output logs to have. Defaults to 2.
    """
    data_products = query_MAST(programID, target_name, extensions, verbose)
    download_from_MAST(data_products, visit_number, outdir, token, verbose)


def query_MAST(programID, target_name, extensions, verbose=2):
    """Queries MAST database to find list of data products related to your observations.

    Args:
        programID (str): ID of the observing program you want to query data
        from. On MAST, referred to as "proposal_ID".
        target_name (str): Name of the target object you want to query data
        from. On MAST, referred to as "target_name".
        extensions (lst of str, optional): File extensions you want to download.
        If None, take all file extensions. Otherwise, take only the files specified.
        Defaults to None.
        verbose (int, optional): From 0 to 2, how much detail you want the
        output logs to have. Defaults to 2.

    Returns:
        lst of str: Filenames to request to download from MAST.
    """
    # Query MAST and get list of relevant data products.
    if verbose >= 1:
        print("Querying MAST for files under program ID {}, target {}...".format(programID, target_name))
    obs_table = Obs.query_criteria(proposal_id=programID, target_name=target_name)
    data_products = Obs.get_product_list(obs_table)

    if extensions:
        data_products = Obs.filter_products(data_products, extension=extensions)
        
    l = [1 for i in data_products if "hst_" not in i["productFilename"]]
    if verbose >= 1:
        print("Found %.0f files " % len(l) + "related to program ID {}, target {}.".format(programID, target_name))

    return data_products


def download_from_MAST(data_products, visit_number, outdir, token=None, verbose=2):
    """From the provided list of data products, downloads only those that have
    the appropriate visit number.

    Args:
        data_products (_type_): output of queryMAST.py.
        visit_number (str): The visit number you want to download, e.g. "01", "02", etc.
        outdir (str): The directory you want the files downloaded to.
        token (str, optional): A MAST authentication token, if you are downloading
        proprietary data. Defaults to None.
        verbose (int, optional): From 0 to 2, how much detail you want the
        output logs to have. Defaults to 2.
    """
    # Creates the output directory if it does not already exist.
    if not os.path.exists(outdir):
        if verbose == 2:
            print("Creating directory {} to store your files in...".format(outdir))
        os.makedirs(outdir)
    if verbose == 2:
        print("Downloaded data will be stored in directory {}.".format(outdir))
    
    if verbose >= 1:
        print("Examining {} queried data products for files in visit number {}...".format(len(data_products),visit_number))

    # Authenticate the user if necessary.
    if token != None:
        if verbose >= 1:
            print("Authenticating MAST query...")
        Obs.login(token=token)
    
    # Download data only if the visit number is correct.
    k = 0
    for data_product in data_products:
        if ("hst_" not in data_product["productFilename"] and data_product["obs_id"][4:6]==visit_number):
            if verbose == 2:
                print("Downloading file number {} from {}...".format(k, data_product["dataURI"]))
            status, msg, url = Obs.download_file(data_product["dataURI"], local_path=os.path.join(outdir, data_product["productFilename"]))
            if verbose == 2:
                print(status, ':', msg)
            k += 1
        else:
            if verbose == 2:
                print("Skipping over file at {}...".format(data_product["dataURI"]))
    if verbose >= 1:
        print("Downloaded {} queried files that had visit number {}.".format(k, visit_number))
    
    # End user authentication session if necessary.
    if token != None:
        Obs.logout()
        if verbose >= 1:
            print("Ended authenticated MAST session.")
