import sys
import pandas as pd
import graphviz
import pydot
from PyQt5 import QtCore, uic, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QLabel, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from generador import generador
from RedNeuronal import RedNeuronal

class App(QMainWindow): 
    def __init__(self):
        super().__init__()
        uic.loadUi('./UI.ui',self)

        #Generador
        self.generarDatosButton.clicked.connect(self.generateDataset)
        self.dataset = []

        #Training
        self.capas = []
        self.trainingTab.setEnabled(False)
        self.entrenarButton.clicked.connect(self.entrenarDataset)
        self.c1CheckTr.clicked.connect(self.activarCapa1Entrenamiento)
        self.c2CheckTr.clicked.connect(self.activarCapa2Entrenamiento)
        self.c2TrBox.setEnabled(False)
        self.c1CheckTr.setChecked(True)
        self.funcTransferenciaC2Tr.setEnabled(False)

        #Ejecucion
        self.ejecucionTab.setEnabled(False)
        self.validarLetraButton.clicked.connect(self.validarLetra)
    
    def generateDataset(self):
        print("hola")
        dataset = generador(self.cantObservaciones.value(),self.maxDistorsion.value(),self.sinDistorsion.value())
        if (dataset):
            self.showAlertGenerador(self.cantObservaciones.value())
            self.trainingTab.setEnabled(True)
    
    def entrenarDataset(self):
        capas = self.setCapas()
        funcTransferencia = self.setFuncTransferencia()
        self.datasetEntrenado = RedNeuronal(
            capas, 
            self.coefAprendizajeTr.value(), 
            self.termMomentoTr.value(), 
            self.precisionTr.value(), 
            self.cantEpocasTr.value(), 
            self.cantObservaciones.value(), 
            funcTransferencia
            ) 
        self.datasetEntrenado.Propagation()
        self.datasetEntrenado.error_patron_v_total.pop(0)
        print("error de test ", (self.datasetEntrenado.error_patron_t_total - self.datasetEntrenado.error_patron_entrenamiento[-1]))
        self.ejecucionTab.setEnabled(True)
        self.showAlertTraining()

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
        print(resultado)
        if resultado[0] > resultado[1] and resultado[0] > resultado[2]:
            print("es una B")
            self.showAlertEjecucion("Es una B")
        elif resultado[1] > resultado[0] and resultado[1] > resultado[2]:
            print("es una D")
            self.showAlertEjecucion("Es una D")
        elif resultado[2] > resultado[0] and resultado[2] > resultado[1]:
            print("es una F")
            self.showAlertEjecucion("Es una F")
        
    def activarCapa1Entrenamiento(self):
        self.c1CheckTr.setChecked(True)
        self.c2TrBox.setEnabled(False)
        self.funcTransferenciaC2Tr.setEnabled(False)
        if (self.c2CheckTr.isChecked()):
            self.c2CheckTr.setChecked(False)

    def activarCapa2Entrenamiento(self):
        self.c2CheckTr.setChecked(True)
        self.c2TrBox.setEnabled(True)
        self.funcTransferenciaC2Tr.setEnabled(True)
        if (self.c1CheckTr.isChecked()):
            self.c1CheckTr.setChecked(False)

    def setCapas(self):
        if (self.c1CheckTr.isChecked()):
            capas = [self.neuronasC1Tr.value()]
        if (self.c2CheckTr.isChecked()):
            capas = [self.neuronasC1Tr.value(),self.neuronasC2Tr.value()]
        print(capas)
        return capas
    
    def setFuncTransferencia(self):
        print(self.funcTransferenciaC1Tr.currentText())
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

    def showAlertGenerador(self, cant):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("AVISO!")
        dlg.setText("Se genero correctamente el dataset de " + str(cant) + " elementos")
        dlg.exec()
    
    def showAlertTraining(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("AVISO!")
        dlg.setText("Se entreno correctamente el dataset")
        dlg.exec()

    def showAlertEjecucion(self, message):
            dlg = QMessageBox(self)
            dlg.setWindowTitle("EXITO!")
            dlg.setText(str(message))
            dlg.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = App()
    GUI.showMaximized()
    sys.exit(app.exec_())
    