
import os

def path_rel_to_file(path, file=__file__):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(file)), path)
