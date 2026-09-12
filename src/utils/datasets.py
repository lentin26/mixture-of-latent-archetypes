
import tempfile
import requests

try:
    import pyreadr
except ImportError:
    pyreadr = None
# Helper function to retrieve data from Github

def download_frac_subtr_to_pandas(file_name):

    valid_file_names = ["fraction.subtraction.data", "fraction.subtraction.qmatrix"]
    if file_name not in valid_file_names:
        raise Exception(f"`file_name` must be one of {valid_file_names}")

    # Path to datasets in Github CDM repo
    path = "https://raw.githubusercontent.com/cran/CDM/master/data/"
    response = requests.get(path + file_name + ".rda")
    response.raise_for_status()

    # Write .RDA file to temp and load using pyreadr
    with tempfile.NamedTemporaryFile(suffix=".rda", delete=True) as f:
        f.write(response.content)
        f.flush()

        result = pyreadr.read_r(f.name)

    # Fetch pandas dataframe
    df = result[file_name]
    return df
    
def download_frac_subtr_X():
    """
    Convert Pandas fraction-subtraction item-response dataset to Numpy.
    """
    return download_frac_subtr_to_pandas("fraction.subtraction.data")\
        .to_numpy()

def download_frac_subtr_Q():
    """
    Convert Pandas fraction-subtraction item-response dataset to Numpy.
    """
    return download_frac_subtr_to_pandas("fraction.subtraction.qmatrix")\
        .to_numpy()