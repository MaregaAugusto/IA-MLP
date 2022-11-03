#datos iniciales
import matplotlib.pyplot as plt
import numpy as np
import csv

import cargarJson as cj
from capa import Capa




precision = 0.05
epocas = 1000

neuronas_entradas = 100
neuronas_ocultas_capa1 = 10
neuronas_ocultas_capa2 = 10
neuronas_salida = 3

n_entrenamiento = 3

# la funcion pesos debe borrar el archivo pesos.json y crear uno nuevo cuando cambian las neuronas de entrada, ocultas y salida
pesos = cj.cargarJson()
if pesos == None:
    cj.definirPesos(neuronas_entradas, neuronas_salida, neuronas_ocultas_capa1, neuronas_ocultas_capa2)
    pesos = cj.cargarJson()

pesos_validacion = None

error_patron_total = list()
error_patron_v_total = [100]
error_patron_t_total = [100]

def entrenar():

    cont_epoca = 0
    error_patron = 100
    while (error_patron >= precision and epocas >= cont_epoca):
        cp = Capa()
        print("Epoca: ", cont_epoca)
        flag = False
        error_patron = 0
        with open("distorsiones700.csv") as csvfile:
            dataset = (csv.reader(csvfile , delimiter=','))
            iteracion = 0
            for row in dataset:
                if flag:
                    fila = int(row.pop(0))
                    clase = row.pop(-1)
                    if clase == "b":
                        salida_deseada = [1,0,0]
                    elif clase == "d":
                        salida_deseada = [0,1,0]
                    elif clase == "f":
                        salida_deseada = [0,0,1]
                    
                    dato_entrenamiento = list(row)
                    #salida_deseada = dato_entrenamiento[-3:]
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

        validar()
        test()
        cont_epoca += 1
        error_patron_total.append(error_patron)
    print("error del patron", error_patron_total)

        





def validar():
    cp = Capa()
    flag = False
    error_patron = 0
    with open("distorsiones200.csv") as csvfile:
        dataset = (csv.reader(csvfile , delimiter=','))
        for row in dataset:
            if flag:
                fila = int(row.pop(0))
                clase = row.pop(-1)
                if clase == "b":
                    salida_deseada = [1,0,0]
                elif clase == "d":
                    salida_deseada = [0,1,0]
                elif clase == "f":
                    salida_deseada = [0,0,1]
                
                dato_entrenamiento = list(row)
                #salida_deseada = dato_entrenamiento[-3:]
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
                
            else:
                flag = True
    if error_patron < error_patron_v_total[-1]:
        pesos_validacion = pesos
        error_patron_v_total.append(error_patron)
        
    
    

def test():
    cp = Capa()
    flag = False
    error_patron = 0
    with open("distorsiones100.csv") as csvfile:
        dataset = (csv.reader(csvfile , delimiter=','))
        iteracion = 0
        for row in dataset:
            if flag:
                fila = int(row.pop(0))
                clase = row.pop(-1)
                if clase == "b":
                    salida_deseada = [1,0,0]
                elif clase == "d":
                    salida_deseada = [0,1,0]
                elif clase == "f":
                    salida_deseada = [0,0,1]
                
                dato_entrenamiento = list(row)
                #salida_deseada = dato_entrenamiento[-3:]
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
                                    
                iteracion += 1
                
            else:
                flag = True
    
    error_patron_t_total.append(error_patron)

def ejecutar(x):
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

entrenar()


plt.plot(error_patron_total)
plt.show()
error_patron_v_total.pop(0)
error_patron_t_total.pop(0)


plt.plot(error_patron_v_total)
plt.show()
plt.plot(error_patron_t_total)
plt.show()

b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

f = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

ejecutar(b)
ejecutar(d)
ejecutar(f)