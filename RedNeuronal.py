#datos iniciales
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
    
    
    def __init__(self, oculta, alpha, beta, precision, epocas, nDataset, is_sigmoidal):
        self.is_sigmoide = is_sigmoidal
        self.oculta = oculta
        self.alpha = alpha
        self.beta = beta
        self.precision = precision
        self.epocas = epocas
        self.pathEntrenamiento = "dataset/"+str(nDataset)+"/entrenamiento.csv"
        self.pathValidacion = "dataset/"+str(nDataset)+"/validacion.csv"
        self.pathTest = "dataset/"+str(nDataset)+"/test.csv"
        self.Pesos = Pesos(self.oculta, nDataset)

    def ForwardPropagationValidacion(self, entradaRed, is_ejecucion = False):
        self.activacion = list()
        self.dActivacion = list()
        self.activacion.append(entradaRed)
        activacion = entradaRed
        for peso in self.Pesos.pesos:
            activacion, dActivacion = self.cp.propagacionAdelante(activacion, self.Pesos.pesosValidacion[peso], self.is_sigmoide[int(peso)])
            self.activacion.append(activacion)
            self.dActivacion.append(dActivacion)
        if is_ejecucion:
            return self.activacion[-1]

    def ForwardPropagation(self, entradaRed, is_ejecucion = False):
        self.activacion = list()
        self.dActivacion = list()
        self.activacion.append(entradaRed)
        activacion = entradaRed
        for peso in self.Pesos.pesos:
            activacion, dActivacion = self.cp.propagacionAdelante(activacion, self.Pesos.pesos[peso], self.is_sigmoide[int(peso)])
            self.activacion.append(activacion)
            self.dActivacion.append(dActivacion)
        if is_ejecucion:
            return self.activacion[-1]
            
    def BackPropagation(self, salidaDeseada, iteracion):
        self.error_neuronas = list()
        aux_pesos = self.Pesos.pesos.copy()
        aux_pesos['0'] = self.activacion[-1]

        aux_dActivacion = self.dActivacion.copy()
        aux_dActivacion.reverse()
        error = salidaDeseada
        flag = True
        coun = len (aux_pesos) - 1
        errores = list()
        for i in aux_dActivacion:
            if flag:
                error = self.cp.ErrorPesosSalida(error, aux_pesos['0'], i)
                flag = False
            else:
                error = self.cp.ErrorPesosOcultas(error, aux_pesos[str(coun)], i)
                coun -= 1
            errores.append(error)
        
        errores.reverse()
        for i in range(len(errores)):
            self.Pesos.pesos[str(i)] = self.cp.modificarPesos(self.Pesos.pesos[str(i)], errores[i], self.alpha, self.activacion[i], iteracion, str(i), self.beta)

        
    
    def Propagation(self):
        cont_epoca = 0
        error_global = 100
        while (error_global >= self.precision and self.epocas >= cont_epoca):
            print("Epoca: ", cont_epoca)
            error_patron, iteracion = self.ejecucion(self.pathEntrenamiento, True)
            self.validar()
            cont_epoca += 1
            self.error_patron_entrenamiento.append(error_patron)
            """ error_global = error_patron/iteracion """
            error_global = error_patron

        self.test()
        print("error del patron", self.error_patron_entrenamiento)

    def ejecucion(self, path, is_entrenamiento):
        self.cp = Capa()
        flag = False
        error_patron = 0
        with open(path) as csvfile:
            dataset = (reader(csvfile , delimiter=','))
            iteracion = 0
            for row in dataset:
                if flag:
                    row.pop(0)
                    clase = row.pop(-1)
                    
                    # list of float
                    entradaRed = [float(i) for i in row]
                    entradaRed =  array(entradaRed)
                    
                    # Forward propagation
                    self.ForwardPropagation(entradaRed)

                    error_patron += self.cp.ErrorPatron(self.salida_deseada[clase], self.activacion[-1])
                    
                    if is_entrenamiento:
                        # backpropagation
                        self.BackPropagation(self.salida_deseada[clase], iteracion)
                    
                    iteracion += 1
                else:
                    flag = True
        return error_patron, iteracion


    def validar(self):
        error_patron, iteracion = self.ejecucion(self.pathValidacion, False)
        
        if (error_patron < self.error_patron_v_total[-1] and self.flag_validacion):
            self.Pesos.setPesosValidacion()
            self.error_patron_v_total.append(error_patron)
        elif (error_patron > self.error_patron_v_total[-1] and self.flag_validacion):
            self.Pesos.guardarJson(True)
            self.flag_validacion = False
            
    def test(self):
        self.error_patron_t_total, iteracion = self.ejecucion(self.pathTest, False)


