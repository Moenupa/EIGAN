import os.path
from glob import glob

par_dir = '/Users/moenupa/Github/EIGAN'

DATA_ROOT = f'{par_dir}/data'
MODEL_ROOT = f'{par_dir}/model'
SCRIPT_ROOT = f'{par_dir}/src'
data_paths, model_paths = glob(f'{DATA_ROOT}/*'), glob(f'{MODEL_ROOT}/*')

assert data_paths
assert model_paths


if __name__ == '__main__':
    from pprint import pprint
    print('data')
    pprint(data_paths)
    print('model')
    pprint(model_paths)
