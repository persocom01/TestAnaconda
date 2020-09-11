# Demonstrates loading and saving in numpy.
import numpy as np
import json

npfile = './datasets/np_save.npy'
jsonfile = './datasets/np_save.json'

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Demonstrates saving and loading a numpy array file.
np.save(npfile, a)
a = np.load(npfile)
print(a)

# Demonstrates saving a numpy array as a json.


class NumpyEncoder(json.JSONEncoder):
    '''
    Special json encoder for numpy types to be passed to
    json.dumps(cls=encoder).
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


j = json.dumps(a, cls=NumpyEncoder)
with open(jsonfile, 'w') as f:
    json.dump(j, f)
