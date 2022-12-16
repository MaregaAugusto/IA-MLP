#datos inicialesejecucion
from numpy import array 
from csv import reader

from Pesos import Pesos
from capa import Capa

class RedNeuronal:
    
    flag_validacion = True
    error_patron_entrenamiento = list()
    error_patron_v_total = [100]
    error_patron_t_total = 0
    salida_deseada = {
        "b": array([1,0,0]),
        "d": array([0,1,0]),
        "f": array([0,0,1])
    }
    
    """
    Inicializar Red neuronal 
    parámetros
        oculta: es una lista con la cantidad de neuronas por capa cada capa oculta 
            ejemplo [2, 3], en la primera capa oculta tenemos 2 neuronas y en la segunda
            3 neuronas 
        alpha: tasa de aprendizaje
        beta: momento
        precision: parámetro de corte de aprendizaje
        epocas: números de época
        nDataset: tamaño del dataset con el que se va a entrenar
        is_sigmoidal: se pasa una lista booleana [True, False] para elegir la función de transición  
    """
    def __init__(self, oculta, alpha, beta, precision, epocas, nDataset, is_sigmoidal):
        self.is_sigmoide = is_sigmoidal
        self.oculta = oculta
        self.alpha = alpha
        self.beta = beta
        self.precision = precision
        self.epocas = epocas
        #Rutas donde se encuentran los dataset
        self.pathEntrenamiento = "dataset/"+str(nDataset)+"/entrenamiento.csv"
        self.pathValidacion = "dataset/"+str(nDataset)+"/validacion.csv"
        self.pathTest = "dataset/"+str(nDataset)+"/test.csv"
        #Inicializar pesos aleatorios o cargar pesos si el modelo ya esta entrenado
        self.Pesos = Pesos(self.oculta, nDataset)

    """
    propagación hacia adelante con los pesos de Validación 
    parámetros
        entradaRed: lista con 0 y 1 que representa una entrada a la Red  

    """
    def ForwardPropagationValidacion(self, entradaRed):
        self.activacion = list()
        self.dActivacion = list()
        activacion = entradaRed
        # bucle de pesos, recorre los pesos de las conecciones de cada capa
        for peso in self.Pesos.pesos:
            activacion = self.cp.propagacionAdelante(activacion, self.Pesos.pesosValidacion[peso], self.is_sigmoide[int(peso)])
        # retorna 
        return activacion

    """
    propagación hacia adelante con los pesos de fin de entrenamiento

    parámetros
        entradaRed: lista con 0 y 1 que representa una entrada a la Red  
        is_ejecucion: False se usa la función para entrenamiento 
                      True se usa la función para una ejecución por la interfaz grafica

    """
    def ForwardPropagation(self, entradaRed, is_ejecucion = False):
        self.activacion = list()
        self.dActivacion = list()
        self.activacion.append(entradaRed)
        activacion = entradaRed
        # bucle de pesos, recorre los pesos de las conecciones de cada capa
        for peso in self.Pesos.pesos:
            # calculamos los f(net) y f'(net)
            # parametros entrada a la capa, pesos correspondientes a la capa, si la capa es sigmoide o lineal
            activacion, dActivacion = self.cp.propagacionAdelante(activacion, self.Pesos.pesos[peso], self.is_sigmoide[int(peso)])
            self.activacion.append(activacion)
            self.dActivacion.append(dActivacion)
        if is_ejecucion:
            return self.activacion[-1]
            
    """
    propagación hacia atras
    parámetros:
        salidaDeseada: valor de la salida deseada es una lista [1, 0, 0] --> b
        interacion: es para saber si ya podemos aplicar el momento

    """        
    def BackPropagation(self, salidaDeseada, iteracion):
        # lista de error de neuronas
        self.error_neuronas = list()
        # copiamos la lista de pesos
        aux_pesos = self.Pesos.pesos.copy()
        # seteamos en la posicion 1 la entrada a la red 
        aux_pesos['0'] = self.activacion[-1]

        # copiamos la lista de derivadas 
        aux_dActivacion = self.dActivacion.copy()
        # revertimos el orden de la lista para que en la primera posicion esten los pesos de salida
        aux_dActivacion.reverse()
        error = salidaDeseada
        flag = True
        coun = len (aux_pesos) - 1
        errores = list()
        # recorremos la lista de derivadas
        for i in aux_dActivacion:
            if flag:
                # la primra vez entra en este bloque y calcula el error de salida 
                error = self.cp.ErrorPesosSalida(error, aux_pesos['0'], i)
                flag = False
            else:
                # para la segunda interación calcula el error de las neurona de capa oculta 
                error = self.cp.ErrorPesosOcultas(error, aux_pesos[str(coun)], i)
                coun -= 1
            errores.append(error)
        # revertimos el orden de la lista de errores
        errores.reverse()
        for i in range(len(errores)):
            # Realizamos la actialización de pesos
            self.Pesos.pesos[str(i)] = self.cp.modificarPesos(self.Pesos.pesos[str(i)], errores[i], self.alpha, self.activacion[i], iteracion, str(i), self.beta)

        
    """
    función de bucle de entrenamiento 
    """
    def Propagation(self):
        cont_epoca = 0
        error_global = 100
        # Bucle de entrenamiento 
        while (error_global >= self.precision and self.epocas >= cont_epoca):
            # reliazamos una ejecucion de una epoca 
            error_patron, iteracion = self.ejecucion(self.pathEntrenamiento, True)
            # realizamos la validación de la red entrenada entrenada con el dataset de validación 
            self.validar()
            cont_epoca += 1
            self.error_patron_entrenamiento.append(error_patron)
            error_global = error_patron/iteracion
        # realizamos la propagación con la red de test
        self.test()

    """
    esta funcion ejecuta una propagación hacia adelante y hacia atras en el caso de que estemos en modo entrenamiento
    parametros:
        path: dataset a utilizar
        is_entrenamiento: si es verdadero hace la propagación hacia atras
    
    """
    def ejecucion(self, path, is_entrenamiento):
        self.cp = Capa()
        flag = False
        error_patron = 0
        with open(path) as csvfile:
            dataset = (reader(csvfile , delimiter=','))
            iteracion = 0
            # recorrer los registros del dataset
            for row in dataset:
                if flag:
                    row.pop(0)
                    clase = row.pop(-1)
                    
                    # list of float
                    entradaRed = [float(i) for i in row]
                    entradaRed =  array(entradaRed)
                    
                    # Forward propagation
                    self.ForwardPropagation(entradaRed)
                    
                    # Calculamos el error del patron
                    error_patron += self.cp.ErrorPatron(self.salida_deseada[clase], self.activacion[-1])
                    
                    if is_entrenamiento:
                        # backpropagation
                        self.BackPropagation(self.salida_deseada[clase], iteracion)
                    
                    iteracion += 1
                else:
                    # la primera iteracion la pasamos por alto porque es el encabezado
                    flag = True
        return error_patron, iteracion

    # función que ejecuta la propagación hacia adelante con el dataset de validación 
    def validar(self):
        error_patron, iteracion = self.ejecucion(self.pathValidacion, False)
        # si el error del patron obtenida es mayor a la anterior se guarda un archivo json con los pesos anteriores
        if (error_patron < self.error_patron_v_total[-1] and self.flag_validacion):
            self.Pesos.setPesosValidacion()
            self.error_patron_v_total.append(error_patron)
        elif (error_patron > self.error_patron_v_total[-1] and self.flag_validacion):
            self.Pesos.guardarJson(True)
            self.flag_validacion = False

    # función que ejecuta la propagación hacia adelante con el dataset de test       
    def test(self):
        self.error_patron_t_total, iteracion = self.ejecucion(self.pathTest, False)


