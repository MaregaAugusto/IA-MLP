from distutils.log import error
from ftplib import error_perm
import numpy as np
import cargarJson as cj

class Capa:

    pesos_anterior = dict()

    #Funcion Sigmoidal
    def sigmoidal(self, x):
        return 1/(np.exp(-x)+1)    

    #Derivada de la funcion sigmoidal
    def d_sigmoidal(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def propagacionAdelante(self, entrada, pesos):
        net= np.dot(entrada, pesos)
        activacion1 = list()
        d_activacion1 = list()
        for i in net:
            activacion1.append(self.sigmoidal(i))
            d_activacion1.append(self.d_sigmoidal(i))

        return net, np.array(activacion1), np.array(d_activacion1)

    def ejecucion(self, entrada, pesos):
        net= np.dot(entrada, pesos)
        activacion1 = list()
        d_activacion1 = list()
        for i in net:
            activacion1.append(self.sigmoidal(i))
            d_activacion1.append(self.d_sigmoidal(i))

        return np.array(activacion1)

    def ErrorPatron(self, salida_deseada, salida_real):
        resta = np.subtract(salida_deseada, salida_real)
        potencia = np.power(resta, 2)
        suma = np.sum(potencia)
        return suma/2


    def ErrorPesosSalida(self, salida_deseada, salida_real, derivada):
        resta = np.subtract(salida_deseada, salida_real)
        multiplicacion = np.multiply(resta, derivada)
        return np.array(multiplicacion)


    def ErrorPesosOcultas(self, error_salida, pesos_salida, derivada):
        matris = error_salida.shape
        error_salida_col = error_salida.reshape(matris[0], 1)
        sumatoria = np.dot(pesos_salida, error_salida_col)
        sumatoria = np.ravel(sumatoria)
        multiplicacion = np.multiply(sumatoria, derivada)
        return multiplicacion

    def modificarPesos(self, pesos, error, alpha, entrada, iteracion, nombre_capa, beta):
        
        n = len(entrada)
        entrada = entrada.reshape(n, 1)
        aux = [error for i in range(n)]
        multiplicar = np.multiply(aux, entrada)
        multiplicar = np.multiply(multiplicar, alpha)
        resultado = np.add(pesos, multiplicar)
        if iteracion < 3:
            self.pesos_anterior[nombre_capa] = pesos
            return resultado
        else:
            resta_pesos = np.subtract(pesos, self.pesos_anterior[nombre_capa])
            momento = np.multiply(resta_pesos, beta)
            resultado = np.add(resultado, momento)
            self.pesos_anterior[nombre_capa] = pesos
            return resultado


