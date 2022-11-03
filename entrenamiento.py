#datos iniciales
import numpy as np
import csv

import cargarJson as cj
from capa import Capa




precision = 0.10
epocas = 100

neuronas_entradas = 100
neuronas_ocultas_capa1 = 5
neuronas_ocultas_capa2 = 5
neuronas_salida = 3

n_entrenamiento = 3

# la funcion pesos debe borrar el archivo pesos.json y crear uno nuevo cuando cambian las neuronas de entrada, ocultas y salida
pesos = cj.cargarJson()
if pesos == None:
    cj.definirPesos(neuronas_entradas, neuronas_salida, neuronas_ocultas_capa1, neuronas_ocultas_capa2)
    pesos = cj.cargarJson()

def entrenar():
    error_global= 100
    cont_epoca = 0
    pesos_anteriores = 0
    while (error_global >= precision and epocas >= cont_epoca):
        cp = Capa()
        print("Epoca: ", cont_epoca)
        flag = False
        error_patron = 0
        with open("entrenamiento.csv") as csvfile:
            dataset = (csv.reader(csvfile , delimiter=';'))
            iteracion = 0
            for row in dataset:
                if flag:
                    dato_entrenamiento = list(row)
                    salida_deseada = dato_entrenamiento[-3:]
                    atributo = dato_entrenamiento[0:100]
                    
                    # list of float
                    atributo = [float(i) for i in atributo]
                    salida_deseada = [float(i) for i in salida_deseada]
                    
                    atributo = np.array(atributo)
                    salida_deseada = np.array(salida_deseada)
                    
                    
                    net1, activacion1, d_activacion1 = cp.propagacionAdelante(atributo, pesos["entrada"])

                    net2, activacion2, d_activacion2 = cp.propagacionAdelante(activacion1, pesos["oculta0"])

                    net3, activacion3, d_activacion3 = cp.propagacionAdelante(activacion2, pesos["salida"])

                    error_patron += cp.ErrorPatron(salida_deseada, activacion3)

                    error_salida = cp.ErrorPesosSalida(salida_deseada, activacion3, d_activacion3)
                    error_oculta = cp.ErrorPesosOcultas(error_salida, pesos["salida"], d_activacion2)
                    error_entrada = cp.ErrorPesosOcultas(error_oculta, pesos["oculta0"], d_activacion1)
                                       
                    alpha = 0.5
                    beta = 0.5

                    pesos["entrada"] = cp.modificarPesos(pesos["entrada"], error_entrada, alpha, atributo, iteracion, "entrada", beta)
                    pesos["oculta0"] = cp.modificarPesos(pesos["oculta0"], error_oculta, alpha, activacion1, iteracion, "oculta0", beta)
                    pesos["salida"] = cp.modificarPesos(pesos["salida"], error_salida, alpha, activacion2, iteracion, "salida", beta)
                    iteracion += 1
                    
                else:
                    flag = True
        cont_epoca += 1
        error_global = error_patron/n_entrenamiento
        





def validar():
    pass

def test():
    pass



entrenar()
x = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
print(x)
cp_ejecutar = Capa()
resultado = cp_ejecutar.ejecucion(np.array(x), pesos["entrada"])
resultado2 = cp_ejecutar.ejecucion(resultado, pesos["oculta0"])
resultado3 = cp_ejecutar.ejecucion(resultado2, pesos["salida"])
print("resultado: ", resultado3)
if resultado3[0] > resultado3[1] and resultado3[0] > resultado3[2]:
    print("es una B")
elif resultado3[1] > resultado3[0] and resultado3[1] > resultado3[2]:
    print("es una D")
elif resultado3[2] > resultado3[0] and resultado3[2] > resultado3[1]:
    print("es una F")