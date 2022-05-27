"""
File utilities
Used to install libraries
Imports only standard python libraries
"""

import os
import sys
import subprocess
import pkg_resources



def os_make_dir(folder):
    """
    Create folder if it does not exist
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def os_path_join(folder, file):
    """remplacement pour `os.path.join(folder, file)` sur windows"""
    return f'{folder}/{file}'


def get_size_str(octets):
    """
    Get size of file in octets
    """
    g_b = round(octets / 2 ** 30, 2)
    m_b = round(octets / 2 ** 20, 2)
    k_b = round(octets / 2 ** 10, 2)
    if g_b > 1:
        ret = f'{g_b} Go'
    elif m_b > 1:
        ret = f'{m_b} Mo'
    elif k_b > 1:
        ret = f'{k_b} ko'
    else:
        ret = f'{octets} octets'
    return ret


def get_filesize(file_path):
    """Taille du fichier"""
    octets = os.stat(file_path).st_size
    return get_size_str(octets)


def install_libraries(required=None) -> None:
    """
    Installer les bibliothèques nécessaires pour ce notebook
    - https://stackoverflow.com/a/44210656/
    """
    if required is None:
        required = {'numpy', 'pandas', 'matplotlib', 'seaborn'}
    # pylint:disable=not-an-iterable
    installed = {pkg.key for pkg in pkg_resources.working_set}  # noqa
    missing = required - installed
    print(f'required modules: {list(required)}')
    print(f'missing modules: {list(missing)}')
    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing],
                              stdout=subprocess.DEVNULL)
