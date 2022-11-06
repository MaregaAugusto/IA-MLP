from numpy import random, sqrt, array, exp, dot, subtract, multiply, sum, power, ravel, add

class Capa:

    pesos_anterior = dict()

    #Funcion Sigmoidal
    def sigmoidal(self, x):
        return 1/(exp(-x)+1)

    #Derivada de la funcion sigmoidal
    def d_sigmoidal(self, x):
        return (exp(-x))/((exp(-x)+1)**2)

    def propagacionAdelante(self, entrada, pesos, is_sigmoidal=True):
        net= dot(entrada, pesos)
        if is_sigmoidal:
            activacion1 = list()
            d_activacion1 = list()
            for i in net:
                activacion1.append(self.sigmoidal(i))
                d_activacion1.append(self.d_sigmoidal(i))
            return array(activacion1), array(d_activacion1)
        else:
            return net, array([1 for i in range(len(net))])

    def ejecucion(self, entrada, pesos):
        net= dot(entrada, pesos)
        activacion1 = list()
        d_activacion1 = list()
        for i in net:
            activacion1.append(self.sigmoidal(i))
            d_activacion1.append(self.d_sigmoidal(i))

        return array(activacion1)

    def ErrorPatron(self, salida_deseada, salida_real):
        resta = subtract(salida_deseada, salida_real)
        potencia = power(resta, 2)
        suma = sum(potencia)
        return suma/2


    def ErrorPesosSalida(self, salida_deseada, salida_real, derivada):
        resta = subtract(salida_deseada, salida_real)
        multiplicacion = multiply(resta, derivada)
        return array(multiplicacion)


    def ErrorPesosOcultas(self, error_salida, pesos_salida, derivada):
        matris = error_salida.shape
        error_salida_col = error_salida.reshape(matris[0], 1)
        sumatoria = dot(pesos_salida, error_salida_col)
        sumatoria = ravel(sumatoria)
        multiplicacion = multiply(sumatoria, derivada)
        return multiplicacion

    def modificarPesos(self, pesos, error, alpha, entrada, iteracion, nombre_capa, beta):

        n = len(entrada)
        entrada = entrada.reshape(n, 1)
        aux = [error for i in range(n)]
        multiplicar = multiply(aux, entrada)
        multiplicar = multiply(multiplicar, alpha)
        resultado = add(pesos, multiplicar)
        if iteracion < 3:
            self.pesos_anterior[nombre_capa] = pesos
            return resultado
        else:
            resta_pesos = subtract(pesos, self.pesos_anterior[nombre_capa])
            momento = multiply(resta_pesos, beta)
            resultado = add(resultado, momento)
            self.pesos_anterior[nombre_capa] = pesos
            return resultado
