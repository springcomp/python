import sys

import numpy as np

def get_max_length_arrays(results: dict) -> int:
    """ Finds the maximum length among the arrays in the specified dictionary """
    max_length = 0
    for arr in results.values():
        if isinstance(arr, (list, np.ndarray)):
            max_length = max(max_length, len(arr))
    return max_length

def has_dict_key(collection: dict, name: str) -> bool:
    """ Returns True if the specified collection has the given key and it is not None """
    return name in collection and collection[name] != None

def pad_arrays_to_length(results: dict, max_length: int) -> None:
    """ Pads the arrays with None to ensure they have the same length """
    for key, arr in results.items():
        if isinstance(arr, (list, np.ndarray)):
            if len(arr) < max_length:
                results[key] += [None] * (max_length - len(arr))

def pad_max_length_arrays(results: dict) -> None:
    """ Pads the arrays with None to ensure they have the same maximum length """
    pad_arrays_to_length(results, get_max_length_arrays(results))

def stderr_exit(error: str) -> None:
    """ Writes to stderr and exits """
    sys.stderr.write(error + '\n')
    sys.exit(1)
