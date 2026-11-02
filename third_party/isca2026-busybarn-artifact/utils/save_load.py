
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))


import pickle


def save_data_to_pkl(data, filename):
    """
    Saves data to a local file.

    Args:
        data: The data to be saved.
        filename: The name of the file where the data will be saved.

    Raises:
        IOError: If the file cannot be opened or written to.
        pickle.PicklingError: If the data cannot be pickled.

    Example:
        save_data_to_file(my_data, 'output.pkl')
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"data saved into {filename}")


def load_data_from_pkl(filename):
    """
    Load data from a local file.

    Args:
        filename (str): The name of the file to read from.

    Returns:
        The data read from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.

    Example:
        data = load_data_from_file('data.pkl')
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"data loader from {filename}")
    return data
