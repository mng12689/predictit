import nbformat
import json
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from nbconvert.preprocessors import ExecutePreprocessor
with open("{dir_path}/Untitled.ipynb".format(dir_path=dir_path)) as f:
    nb = nbformat.read(f, as_version=4)
ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
ep.preprocess(nb,{'metadata': {'path': './'}})

with open("{dir_path}/Untitled.nbconvert.ipynb".format(dir_path=dir_path)) as f:
    data = json.load(f)
    for c in data["cells"]:
        if "outputs" in c and len(c["outputs"]) > 0:
            for output in c["outputs"]:
                for t in output["text"]:
                    print(t, end='')
