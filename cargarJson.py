import json
import numpy as np
from random import random 

#json
#   entrada
#   oculta
#   salida
def cargarJson():
    try:
        with open('pesos.json') as file:
            pesos = json.load(file)
            for i in pesos:
                aux = list()
                for j in range(len(pesos[i])):
                    aux.append(pesos[i][str(j)])
                pesos[i] = np.array(aux)
            return pesos
    except FileNotFoundError:
        return None

def ramdomPesos(x, y):
    pesos = dict()
    pesos = np.random.uniform(0.,1.,size=(x,y))
    aux = dict()
    for i in range(len(pesos)):
        aux[i] = list(pesos[i])
    return aux

def guardarJson(pesos):
    with open('pesos.json', 'w') as file:
        json.dump(pesos, file)

def Modificar(pesos):
    for i in pesos:
        aux = dict()
        for j in range(len(pesos[i])):
            aux[j] = list(pesos[i][j])
        pesos[i] = aux
    guardarJson(pesos)
    


def definirPesos(entrada, salida, *oculta):
    pesos = dict()
    for i in range(len(oculta)):
        if i == 0:
            pesos['entrada'] = ramdomPesos(entrada, oculta[i])
            
        if i == len(oculta) - 1:
            pesos['salida'] = ramdomPesos(oculta[i], salida)
        else:
            pesos['oculta'+str(i)] = ramdomPesos(oculta[i-1], oculta[i])

    guardarJson(pesos)

