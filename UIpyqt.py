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

        #Ejecucion
        self.ejecucionTab.setEnabled(False)
    
    def generateDataset(self):
        dataset = generador(self.cantObservaciones.value(),self.maxDistorsion.value(),self.sinDistorsion.value())
        if (dataset):
            self.showAlertGenerador(self.cantObservaciones.value())
            self.trainingTab.setEnabled(True)
    
    def entrenarDataset(self):
        print('hola')
        capas = self.setCapas()
        self.datasetEntrenado = RedNeuronal(
            capas, 
            self.coefAprendizajeTr.value(), 
            self.termMomentoTr.value(), 
            self.precisionTr.value(), 
            self.cantEpocasTr.value(), 
            self.cantObservaciones.value(), 
            [True, True, True]
            ) 
        self.datasetEntrenado.Propagation()
        self.datasetEntrenado.error_patron_v_total.pop(0)
        print("error de test ", (self.datasetEntrenado.error_patron_t_total - self.datasetEntrenado.error_patron_entrenamiento[-1]))

        #(self, oculta, alpha, beta, precision, epocas, nDataset, is_sigmoidal)
        
    def activarCapa1Entrenamiento(self):
        self.c1CheckTr.setChecked(True)
        self.c2TrBox.setEnabled(False)
        if (self.c2CheckTr.isChecked()):
            self.c2CheckTr.setChecked(False)
       
    
    def activarCapa2Entrenamiento(self):
        self.c2CheckTr.setChecked(True)
        self.c2TrBox.setEnabled(True)
        if (self.c1CheckTr.isChecked()):
            self.c1CheckTr.setChecked(False)

    def setCapas(self):
        if (self.c1CheckTr.isChecked()):
            capas = [self.neuronasC1Tr.value()]
        if (self.c2CheckTr.isChecked()):
            capas = [self.neuronasC1Tr.value(),self.neuronasC2Tr.value()]
        print(capas)
        return capas

    def showAlertGenerador(self, cant):
            dlg = QMessageBox(self)
            dlg.setWindowTitle("AVISO!")
            dlg.setText("Se genero correctamente el dataset de " + str(cant) + " elementos")
            dlg.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = App()
    GUI.showMaximized()
    sys.exit(app.exec_())
    