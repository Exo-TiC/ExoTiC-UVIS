import shlex
import numpy as np

def parse_config(path_to_config_file):
    '''
    Parses config files to create a dictionary of inputs.

    :param path_to_config_file: str. Path to the .hustle file that is being read.
    :return: dict of instructions for run_pipeline to follow.
    '''
    # Open the dictionary.
    config ={}

    # Read out all lines.
    with open(path_to_config_file,mode='r') as f:
        lines = f.readlines()

    # Process all lines.
    for line in lines:
        line = shlex.split(line, posix=False)
        # Check if it is empty line or comment line and pass.
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        key = line[0]
        param = line[1]
        try:
            param = eval(param)
        except:
            pass
        config[key] = param

    return config