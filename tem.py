import json
import numpy as np

with open('pesos.json') as file:
            pesos = json.load(file)

for i in pesos:
    aux = list()
    for j in range(len(pesos[i])):
        aux.append(pesos[i][str(j)])
    pesos[i] = np.array(aux)

print(pesos["salida"])