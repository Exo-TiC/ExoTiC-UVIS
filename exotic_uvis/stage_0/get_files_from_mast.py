import os

from astroquery.mast import Observations as Obs

def do(programID, target_name, visit_number, outdir, extensions):
    '''
    Queries MAST database and downloads *flt and *spt fits files from specified program, target, and visit number.
    
    :param programID: str. ID of the observing program you want to query data from On MAST, referred to as "proposal_ID".
    :param target_name: str. Name of the target object you want to query data from. On MAST, referred to as "target_name".
    :param visit_number: str. The visit number you want to download, e.g. "01", "02", etc.
    :param outdir: str. The directory you want the files downloaded to.
    :param extensions: lst of str or None. File extensions you want to download. If None, take all file extensions. Otherwise, take only the files specified.
    :return: files downloaded into the specified directory. No callables.
    '''
    data_products = query_MAST(programID, target_name, extensions)
    download_from_MAST(data_products, visit_number, outdir)

def query_MAST(programID, target_name, extensions):
    '''
    Queries MAST database to find list of data products related to your observations.
    
    :param programID: str. ID of the observing program you want to query data from On MAST, referred to as "proposal_ID".
    :param target_name: str. Name of the target object you want to query data from. On MAST, referred to as "target_name".
    :param extensions: lst of str or None. File extensions you want to download. If None, take all file extensions. Otherwise, take only the files specified.
    :return: list of str filenames to request to download from MAST.
    ''' 
    # Query MAST and get list of relevant data products.
    print("Querying MAST for files under program ID {}, target {}...".format(programID, target_name))
    obs_table = Obs.query_criteria(proposal_id=programID, target_name=target_name)
    data_products = Obs.get_product_list(obs_table)
    if extensions:
        data_products = Obs.filter_products(data_products, extension=extensions)
    l = [1 for i in data_products if "hst_" not in i["productFilename"]]
    print("Found %.0f files " % len(l) + "related to program ID {}, target {}.".format(programID, target_name))

    return data_products

def download_from_MAST(data_products, visit_number, outdir):
    '''
    From the provided list of data products, downloads only those that have the appropriate visit number.

    :param data_products: output of queryMAST.py.
    :param visit_number: str. The visit number you want to download, e.g. "01", "02", etc.
    :param outdir: str. The directory you want the files downloaded to.
    :return: files downloaded into the specified directory. No callables.
    '''
    # Creates the output directory if it does not already exist.
    if not os.path.exists(outdir):
        print("Creating directory {} to store your files in...".format(outdir))
        os.makedirs(outdir)
    print("Downloaded data will be stored in directory {}.".format(outdir))
    
    print("Examining {} queried data products for files in visit number {}...".format(len(data_products),visit_number))
    # Download data only if the visit number is correct.
    k = 0
    for data_product in data_products:
        if ("hst_" not in data_product["productFilename"] and data_product["obs_id"][4:6]==visit_number):
            print("Downloading file number {} from {}...".format(k, data_product["dataURI"]))
            Obs.download_file(data_product["dataURI"], local_path=os.path.join(outdir, data_product["productFilename"]))
            k += 1
        else:
            print("Skipping over file at {}...".format(data_product["dataURI"]))
    print("Downloaded {} queried files that had visit number {}.".format(k, visit_number))