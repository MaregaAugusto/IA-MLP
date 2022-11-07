
#@markdown Instalamos las librerias necesarias
#!pip install pandas Faker
import pandas as pd
from faker import Faker
import random
import copy
import numpy as np

#@markdown Definimos las letras B, D y F
letraB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "b"]
letraD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "d"]
letraF = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "f"]

letras = [letraB, letraD, letraF]

porcentajeValidacion = 20
porcentajeTest = 10
porcentajeTraining = 70

#@markdown Generador.
def generador(cantObs,maxDistorsion,sinDistorsion):
  result = []
  fake = Faker()
  sinDist = (cantObs*sinDistorsion)/100
  # Vamos a generar una letra de cada una para tener aproximadamente la misma cantidad de cada una

  selectorLetra = 0

  # 0 a Cantidad de letras generadas
  for obs in range(0, cantObs):
    if sinDist > 0:
      distorsion = 0
      sinDist = sinDist - 1
    else:
      # Aleatorizamos la distorsión entre 1 y 30 (porque aca se hacen solo los distorcionados)
      distorsion = random.randint(1,maxDistorsion + 1)

    # Elegimos las letras del array de letras en orden
    letra = copy.deepcopy(letras[selectorLetra])
    # Bucle para reemplazar tantos bits como sea nuestra distorsión
    contDist = 0;
    for i in range(0, distorsion):
      # Seleccionamos un bit entre 100
      distX = random.randint(1, 100)
  # Cambio de digito segun corresponda
      if letra[distX] == 0:
        letra[distX] = 1;
      elif letra[distX] == 1:
        letra[distX] = 0;

    result.append(letra);
    # Pasamos a la siguiente letra
    selectorLetra = selectorLetra + 1
    if selectorLetra > (len(letras) - 1):
      selectorLetra = 0
  
  # desordenamos result
  result = random.sample(result,len(result))

  cantTraining = int((cantObs * porcentajeTraining)/100)
  cantTest = int((cantObs * porcentajeTest)/100)
  cantValidacion = int((cantObs * porcentajeValidacion)/100)

  result_training = result[0:cantTraining]
  result_test = result[(cantTraining):(cantTraining + cantTest)]
  result_validacion = result[(cantTraining + cantTest):(cantObs)]

  # Una vez que finalizamos el bucle, cargamos este arreglo de salida a un .csv 

  letras_distorsionadas_training = pd.DataFrame(result_training);
  letras_distorsionadas_training.to_csv(r'dataset\\entrenamiento' + '.csv');

  letras_distorsionadas_test = pd.DataFrame(result_test);
  letras_distorsionadas_test.to_csv(r'dataset\\test' + '.csv');

  letras_distorsionadas_validacion = pd.DataFrame(result_validacion);
  letras_distorsionadas_validacion.to_csv(r'dataset\\validacion' + '.csv');

  return result