import pdb
import numpy as np
from os import listdir
from os.path import isfile, join

files = [f for f in listdir('./') if isfile(join('./', f))]

output = {}
for f in files:
    if 'div-img-rand' in f:
        file = open(f, 'r')
        while True:
            line = file.readline()
            if not line:
                break
            if "{'disagreement':" in line:
                tokens = line.split(' ')
                div = []
                for token in tokens:
                    if ',' in token or '}' in token:
                        div.append(float(token[:-3]))
                break
        output[f] = div
        print(f, div)
        file.close()

# print(output)
