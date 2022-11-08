from json import load, dump
from numpy import random, sqrt, array
from os import mkdir
from os.path import exists

def ramdomPesos(x, y):
    pesos = dict()
    pesos = random.uniform(-1.,1.,size=(x,y))/sqrt(x*y)
    aux = dict()
    for i in range(len(pesos)):
        aux[i] = list(pesos[i])
    return aux

def JsontoArray(pesos):
    for i in pesos:
        aux = list()
        for j in range(len(pesos[i])):
            aux.append(pesos[i][str(j)])
        pesos[i] = array(aux)
    return pesos

def DicttoArray1(pesos):
    for i in pesos:
        aux = list()
        for j in range(len(pesos[i])):
            aux.append(pesos[i][j])
        pesos[i] = array(aux)
    return pesos

def ArraytoDict(pesos):
    for i in pesos:
        aux = dict()
        for j in range(len(pesos[i])):
            aux[j] = list(pesos[i][j])
        pesos[i] = aux
    return pesos

class Pesos:
    entrada = 100
    salida = 3
    

    def __init__(self, oculta, nDataset):
        self.oculta = oculta
        self.setPath(str(nDataset))
        self.cargarPesos()

    def setPesosValidacion(self):
        self.pesosValidacion = self.pesos

    def setPath(self, nDataset):
        path = 'pesos'
        if not exists(path):
            mkdir(path)
        path = 'pesos\\'+nDataset
        if not exists(path):
            mkdir(path)
        path = path+"\\"+str(self.entrada)+"-"
        for i in self.oculta:
            path = path + str(i)+"-"
        self.path = path + str(self.salida) + '.json'
        self.pathValidacion = path + str(self.salida) + 'Validacion.json'

    def cargarPesos(self):
        if exists(self.path):
            with open(self.path, 'r') as file_json:
                self.pesos = load(file_json)
            self.pesos = JsontoArray(self.pesos)
        else:
            self.definirPesos()

        if exists(self.pathValidacion):
            with open(self.pathValidacion, 'r') as file_json:
                self.pesosValidacion = load(file_json)
            self.pesosValidacion = JsontoArray(self.pesosValidacion)

    def definirPesos(self):
        self.pesos = dict()
        self.pesos['0'] = ramdomPesos(self.entrada, self.oculta[0])
        x=0
        for i in range((len(self.oculta)-1)):
            self.pesos[str(i+1)] = ramdomPesos(self.oculta[i], self.oculta[i+1])
            x = i+1 
        self.pesos[str(x+1)] = ramdomPesos(self.oculta[-1], self.salida)
        self.guardarJson()

    def guardarJson(self, validacion = False):
        if validacion:
            self.pesosValidacion = ArraytoDict(self.pesosValidacion)
            with open(self.pathValidacion, 'w') as file_json:
                dump(self.pesosValidacion, file_json)
            self.pesosValidacion = DicttoArray1(self.pesosValidacion)

        else:
            self.pesos = ArraytoDict(self.pesos)
            with open(self.path, 'w') as file_json:
                dump(self.pesos, file_json)
            self.pesos = DicttoArray1(self.pesos)

