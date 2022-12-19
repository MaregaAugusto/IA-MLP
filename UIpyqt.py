from sys import argv, exit
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from generador import generador
from RedNeuronal import RedNeuronal
from matplotlib.pyplot import plot, show, figure, title, xlabel, ylabel, legend

class App(QMainWindow): 
    def __init__(self):
        super().__init__()
        loadUi('./UI.ui',self)

        #Variables Tab Generador
        self.generarDatosButton.clicked.connect(self.generateDataset)
        self.dataset = []

        #Variables Tab Training
        self.capas = []
        self.trainingTab.setEnabled(False)
        self.entrenarButton.clicked.connect(self.entrenarDataset)
        self.c1CheckTr.clicked.connect(self.activarCapa1Entrenamiento)
        self.c2CheckTr.clicked.connect(self.activarCapa2Entrenamiento)
        self.c2TrBox.setEnabled(False)
        self.c1CheckTr.setChecked(True)
        self.funcTransferenciaC2Tr.setEnabled(False)

        #Variables Tab Ejecucion
        self.ejecucionTab.setEnabled(False)
        self.validarLetraButton.clicked.connect(self.validarLetra)
    
    # Parametros: cantidad de observaciones, maxima distorsion y % sin distorsion
    # Generamos el dataset y lo guardamos, en caso de exito habilitamos la tab de entrenamiento
    def generateDataset(self):
        dataset = generador(int(self.cantObservaciones.currentText()))
        if (dataset):
            self.showAlertGenerador(int(self.cantObservaciones.currentText()))
            self.trainingTab.setEnabled(True)
    
    # Parametros: capas, coeficiente de entrenamiento, termino de momento, presicion, cantidad de epocas, cantidad de observaciones, funcion de transferencia
    # Entrenamos el dataset generado anteriormente y habilitamos la tab de validacion
    def entrenarDataset(self):
        capas = self.setCapas()
        funcTransferencia = self.setFuncTransferencia()
        self.datasetEntrenado = RedNeuronal(
            capas, 
            self.coefAprendizajeTr.value(), 
            self.termMomentoTr.value(), 
            self.precisionTr.value(), 
            self.cantEpocasTr.value(), 
            int(self.cantObservaciones.currentText()), 
            funcTransferencia
            ) 
        Presicion = self.datasetEntrenado.Propagation()
        self.datasetEntrenado.error_patron_v_total.pop(0)
        plot(self.datasetEntrenado.error_patron_entrenamiento, marker='o', color='blue', label='Entrenamiento')
        plot(self.datasetEntrenado.error_patron_v_total, color='red', label='Validacion')
        xlabel('Epocas')
        ylabel('Error de entrenamiento')
        title('MSE')
        legend()
        show()
        self.ejecucionTab.setEnabled(True)
        self.showAlertTraining( Presicion)

    # Parametros: check de botones que forman la letra ingresada
    # Validamos el input ingresado mediante la cuadricula 10x10 con botones, realizamos la propagacion hacia adelante y vemos a que letra se asemeja segun la red entrenada
    def validarLetra(self):
        letra = [
            self.button1.isChecked(), self.button2.isChecked(), self.button3.isChecked(), self.button4.isChecked(), self.button5.isChecked(), self.button6.isChecked(), self.button7.isChecked(), self.button8.isChecked(), self.button9.isChecked(), self.button10.isChecked(),
            self.button1_2.isChecked(), self.button2_2.isChecked(), self.button3_2.isChecked(), self.button4_2.isChecked(), self.button5_2.isChecked(), self.button6_2.isChecked(), self.button7_2.isChecked(), self.button8_2.isChecked(), self.button9_2.isChecked(), self.button10_2.isChecked(),
            self.button1_3.isChecked(), self.button2_3.isChecked(), self.button3_3.isChecked(), self.button4_3.isChecked(), self.button5_3.isChecked(), self.button6_3.isChecked(), self.button7_3.isChecked(), self.button8_3.isChecked(), self.button9_3.isChecked(), self.button10_3.isChecked(),
            self.button1_4.isChecked(), self.button2_4.isChecked(), self.button3_4.isChecked(), self.button4_4.isChecked(), self.button5_4.isChecked(), self.button6_4.isChecked(), self.button7_4.isChecked(), self.button8_4.isChecked(), self.button9_4.isChecked(), self.button10_4.isChecked(),
            self.button1_5.isChecked(), self.button2_5.isChecked(), self.button3_5.isChecked(), self.button4_5.isChecked(), self.button5_5.isChecked(), self.button6_5.isChecked(), self.button7_5.isChecked(), self.button8_5.isChecked(), self.button9_5.isChecked(), self.button10_5.isChecked(),
            self.button1_6.isChecked(), self.button2_6.isChecked(), self.button3_6.isChecked(), self.button4_6.isChecked(), self.button5_6.isChecked(), self.button6_6.isChecked(), self.button7_6.isChecked(), self.button8_6.isChecked(), self.button9_6.isChecked(), self.button10_6.isChecked(),
            self.button1_7.isChecked(), self.button2_7.isChecked(), self.button3_7.isChecked(), self.button4_7.isChecked(), self.button5_7.isChecked(), self.button6_7.isChecked(), self.button7_7.isChecked(), self.button8_7.isChecked(), self.button9_7.isChecked(), self.button10_7.isChecked(),
            self.button1_8.isChecked(), self.button2_8.isChecked(), self.button3_8.isChecked(), self.button4_8.isChecked(), self.button5_8.isChecked(), self.button6_8.isChecked(), self.button7_8.isChecked(), self.button8_8.isChecked(), self.button9_8.isChecked(), self.button10_8.isChecked(),
            self.button1_9.isChecked(), self.button2_9.isChecked(), self.button3_9.isChecked(), self.button4_9.isChecked(), self.button5_9.isChecked(), self.button6_9.isChecked(), self.button7_9.isChecked(), self.button8_9.isChecked(), self.button9_9.isChecked(), self.button10_9.isChecked(),
            self.button1_10.isChecked(), self.button2_10.isChecked(), self.button3_10.isChecked(), self.button4_10.isChecked(), self.button5_10.isChecked(), self.button6_10.isChecked(), self.button7_10.isChecked(), self.button8_10.isChecked(), self.button9_10.isChecked(), self.button10_10.isChecked(),
        ]
        resultado = self.datasetEntrenado.ForwardPropagation(letra, True)
        if resultado[0] > resultado[1] and resultado[0] > resultado[2]:
            self.showAlertEjecucion("Es una B ", resultado)
        elif resultado[1] > resultado[0] and resultado[1] > resultado[2]:
            self.showAlertEjecucion("Es una D ", resultado)
        elif resultado[2] > resultado[0] and resultado[2] > resultado[1]:
            self.showAlertEjecucion("Es una F ", resultado)

    # Funcion que controla el comportamiento de activacion/desactivacion de los inputs de capas segun se seleccione la cantidad en la etapa de entremiento (1 o 2 capas)    
    def activarCapa1Entrenamiento(self):
        self.c1CheckTr.setChecked(True)
        self.c2TrBox.setEnabled(False)
        self.funcTransferenciaC2Tr.setEnabled(False)
        if (self.c2CheckTr.isChecked()):
            self.c2CheckTr.setChecked(False)

    # Funcion que controla el comportamiento de activacion/desactivacion de los inputs de capas segun se seleccione la cantidad en la etapa de entremiento (1 o 2 capas)
    def activarCapa2Entrenamiento(self):
        self.c2CheckTr.setChecked(True)
        self.c2TrBox.setEnabled(True)
        self.funcTransferenciaC2Tr.setEnabled(True)
        if (self.c1CheckTr.isChecked()):
            self.c1CheckTr.setChecked(False)

    # Genera un array con la cantidad de capas seleccionadas en la etapa de entrenamiento
    def setCapas(self):
        if (self.c1CheckTr.isChecked()):
            capas = [self.neuronasC1Tr.value()]
        if (self.c2CheckTr.isChecked()):
            capas = [self.neuronasC1Tr.value(),self.neuronasC2Tr.value()]
        return capas

    # Genera un array con las funciones de transferencia seleccionadas en la etapa de entrenamiento
    def setFuncTransferencia(self):
        funcTransferencia = []
        if(self.funcTransferenciaC1Tr.currentText() == "Sigmoidal"):
            funcTransferencia.append(True)
        else:
            funcTransferencia.append(False)
        if (self.c2CheckTr.isChecked()):
            if(self.funcTransferenciaC2Tr.currentText() == "Sigmoidal"):
                funcTransferencia.append(True)
            else:
                funcTransferencia.append(False)
        if(self.funcTransferenciaSalTr.currentText() == "Sigmoidal"):
            funcTransferencia.append(True)
        else:
            funcTransferencia.append(False)
        return funcTransferencia

    # Genera una alerta cuando se genera correctamente el dataset
    def showAlertGenerador(self, cant):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("AVISO!")
        dlg.setText("Se genero correctamente el dataset de " + str(cant) + " elementos")
        dlg.exec()
    
    # Genera una alerta cuando se entrena correctamente el dataset
    def showAlertTraining(self, Presicion):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("AVISO!")
        mensaje = "La red neuronal se entreno correctamente \nCon una presición: "+ str(Presicion)
        dlg.setText(mensaje)
        dlg.exec()

    # Genera una alerta cuando se ejecuta correctamente y predice una letra
    def showAlertEjecucion(self, message, result):
            letra = {
                'B': result[0],
                'D': result[1],
                'F': result[2] 
            }
            for clave, valor in letra.items():
                message += "\nPorcentaje " + clave + ": {0:.2f}".format((valor*100))+"%"
            dlg = QMessageBox(self)
            dlg.setWindowTitle("EXITO!")
            dlg.setText(str(message))
            dlg.exec()

#Funcion main de la interface
if __name__ == '__main__':
    app = QApplication(argv)
    GUI = App()
    GUI.showMaximized()
    exit(app.exec_())
    