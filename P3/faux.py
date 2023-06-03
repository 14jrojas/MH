import sys
import numpy as np

def leer_datos(ruta_fichero):
    leyendo = False
    nombre_atributos = []
    datos = []
    clases = []
    with open(ruta_fichero,'r') as fichero:
        lineas = fichero.readlines()
        for linea in lineas:
            if linea.startswith('%') or linea.startswith('\n'):
                continue
            if linea.startswith('@relation'):
                continue
            if linea.startswith('@attribute'):
                nombre_atributos.append(linea.split(' ')[1])
            if linea.startswith('@data'):
                leyendo = True
                continue

            if leyendo:
                datos_separados = linea.split('\n')[0].split(',')
                n = len(datos_separados)
                datos.append(datos_separados[:n-1])
                clases.append(datos_separados[n-1:])

    return nombre_atributos, datos, clases

def normalizar_datos(datos):
    datos = (datos - np.min(datos, axis=0)) / (np.max(datos, axis=0) - np.min(datos, axis=0))
    return datos

def clasificador1nn(pesos, dato, datos, clases, indice=-1):
    distancias = pesos*(datos-dato)**2
    distancias = np.sum(distancias, axis=1)
    distancias = np.sqrt(distancias)
    if indice != -1:
        distancias[indice] = np.inf
    return clases[np.argmin(distancias)]

def tasa_clas(pesos, datos_clasificar, clases_clasificar, datos_train, clases_train):
    if(datos_clasificar.shape[0] == datos_train.shape[0]):
        clases_predecidas = np.array([clasificador1nn(pesos,dato,datos_train,clases_train,i) for i,dato in enumerate(datos_clasificar)])
    else:
        clases_predecidas = np.array([clasificador1nn(pesos,dato,datos_train,clases_train) for dato in datos_clasificar])
        
    return 100 * len(np.where(clases_predecidas == clases_clasificar)[0]) / len(clases_clasificar)

def tasa_red(pesos):
    return 100 * len(pesos[pesos < 0.1]) / len(pesos)

def tasa_fitness(alpha, pesos, datos_clasificar, clases_clasificar, datos_train, clases_train):
    return alpha * tasa_clas(pesos, datos_clasificar, clases_clasificar, datos_train, clases_train) + (1-alpha) * tasa_red(pesos)




