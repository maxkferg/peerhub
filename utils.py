import os

COMBINED = "combined"

def list_directories(path):
    dirs = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    if COMBINED in dirs:
    	dirs.remove(COMBINED)
    dirs = [os.path.join(path, d) for d in dirs]
    return dirs#sorted(dirs)


def invert_map(map):
	return {v: k for k, v in map.items()}