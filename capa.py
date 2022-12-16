from numpy import random, sqrt, array, exp, dot, subtract, multiply, sum, power, ravel, add

class Capa:

    pesos_anterior = dict()

    #Funcion Sigmoidal
    def sigmoidal(self, x):
        return 1/(exp(-x)+1)

    #Derivada de la funcion sigmoidal
    def d_sigmoidal(self, x):
        return (x*(1-x))
        #return (exp(-x))/((exp(-x)+1)**2)

    """
    funcion que calcula la propagación hacia adelante de una capa
    parametros:
        entrada: lista con los valores de entrada a la red o f(net) de la capas previas 
        pesos: los pesos de las conexiones entre la entrada y la neurona (matriz)
        is_sigmoidal: True usamos funcion de transferencia sigmoidal
    """
    def propagacionAdelante(self, entrada, pesos, is_sigmoidal=True):
        # relizamos una multiplicación de matrices para obtener los net de cada neurona 
        net= dot(entrada, pesos)
        if is_sigmoidal:
            activacion1 = list()
            d_activacion1 = list()
            for i in net:
                # calculamos la funcion de transferencia y su derivada 
                f_net = self.sigmoidal(i)
                activacion1.append(f_net)
                d_activacion1.append(self.d_sigmoidal(f_net))
            return array(activacion1), array(d_activacion1)
        else:
            # en el caso que la funcion sea lineal 
            return net, array([1 for i in range(len(net))])

    """ def ejecucion(self, entrada, pesos):
        net= dot(entrada, pesos)
        activacion1 = list()
        d_activacion1 = list()
        for i in net:
            activacion1.append(self.sigmoidal(i))
            d_activacion1.append(self.d_sigmoidal(i))

        return array(activacion1) """

    """
        función que calcula el error del patron
        parámetros:
            salida_deseada: es una lista [1,0,0] --> b
            salida_real: es la obtenida por la red
    """   
    def ErrorPatron(self, salida_deseada, salida_real):
        # realizamos la resta de las salidas 
        resta = subtract(salida_deseada, salida_real)
        # elevamos al cuadrado 
        potencia = power(resta, 2)
        # sumamos todas la salidas 
        suma = sum(potencia)
        # divimos por 2
        return suma/2

    """
        función que calcula el error del las neuronas de salida
        parámetros:
            salida_deseada: es una lista [1,0,0] --> b
            salida_real: es la obtenida por la red
            derivada: de la neurona a calcular el error 
            
            delta = (saida_deseada - salida_obtenida) * f'(net)
    """  

    def ErrorPesosSalida(self, salida_deseada, salida_real, derivada):
        # realizamos la resta de las salidas
        resta = subtract(salida_deseada, salida_real)
        # multiplicamos la derivada
        multiplicacion = multiply(resta, derivada)
        return array(multiplicacion)

    """
        función que calcula el error del las neuronas de oculta
        parámetros:
            error_capa_superior: es el error de la capa superior 
            pesos: son los pesos de las conexiones con la neurona de la capa superior 
            derivada: de la neurona a calcular el error 
            
            delta = f'(net) * (pesos de)
    """      
    def ErrorPesosOcultas(self, error_capa_superior, pesos, derivada):
        # obtenemos las filas y columnas de la matriz
        matriz = error_capa_superior.shape
        # esta funcion nos genera un matriz con una sola columna y un elemento por fila  
        error_capa_superior_col = error_capa_superior.reshape(matriz[0], 1)
        # realizamos la multiplicació de los pesos y la entrada
        sumatoria = dot(pesos, error_capa_superior_col)
        # revel nos vuelve a crear un matriz de un fila y n columnas 
        sumatoria = ravel(sumatoria)
        # multiplicamos el resultado de la sumatoria por su correspondiente f'(net)
        multiplicacion = multiply(sumatoria, derivada)
        return multiplicacion

    """
    Esta función modifica los pasos de la red
    parámetros:
        pesos: pesos actuales de las conexiones a modificar
        error: error de la neurona de la capa superior 
        alpha: tas de aprendizaje
        entrada: entrada a la red o f(net)
        interacion: iteracion la usamos para saber si tenemos que aplicar el momento
        nombre_capa: la usamos para guardar el peso anterior para la proxima iteracion
        beta: momento 
    """

    def modificarPesos(self, pesos, error, alpha, entrada, iteracion, nombre_capa, beta):
        
        n = len(entrada)
        # esta funcion nos genera un matriz con una sola columna y un elemento por fila  
        entrada = entrada.reshape(n, 1)
        # genero una matriz de n columnas y en cada fila los errores de la capa_superior 
        aux = [error for i in range(n)]
        # multiplicamos la entrada a la neurona 
        multiplicar = multiply(aux, entrada)
        # multiplicamos el resultado anterior por la tasa de aprendizaje 
        multiplicar = multiply(multiplicar, alpha)
        # sumamos el resultado anterior por los pesos en el tiempo T
        resultado = add(pesos, multiplicar)
        if iteracion < 3:
            self.pesos_anterior[nombre_capa] = pesos
            return resultado
        else:
            # agregamos el calculo del momento despues de la 3 iteración
            # restamos los pesos de t-1 a ls pesos actuaes 
            resta_pesos = subtract(pesos, self.pesos_anterior[nombre_capa])
            # luego multiplicamos por el momento
            momento = multiply(resta_pesos, beta)
            # sumamos 
            resultado = add(resultado, momento)
            self.pesos_anterior[nombre_capa] = pesos
            return resultado
