import sys
import numpy as np
import timeit
from tabulate import tabulate

import faux


# FUNCIONES NECESARIAS PARA ALGORITMO BUSQUEDA LOCAL

def mov(pesos, indice, delta):
    pesos[indice] = pesos[indice] + np.random.normal(loc = 0, scale = delta)
    if pesos[indice] > 1: pesos[indice] = 1
    if pesos[indice] < 0: pesos[indice] = 0

def busqueda_local(n_max_vecinos, n_max_evaluaciones, delta, datos, clases):
    pesos = np.random.uniform(0, 1, datos.shape[1])
    indices = np.arange(len(pesos))

    tasa_ultima_mejora = faux.tasa_fitness(0.8, pesos, datos, clases, datos, clases)

    vecinos_consecutivos = 0
    evaluaciones = 0
    while(vecinos_consecutivos < n_max_vecinos and evaluaciones < n_max_evaluaciones):
        np.random.shuffle(indices)
        for indice in indices:
            copia = np.copy(pesos)
            mov(copia, indice, delta=delta)
            vecinos_consecutivos = vecinos_consecutivos + 1
            if vecinos_consecutivos >= n_max_vecinos: break
            tasa_actual = faux.tasa_fitness(0.8, copia, datos, clases, datos, clases)
            evaluaciones = evaluaciones + 1
            if (tasa_actual > tasa_ultima_mejora):
                pesos = copia
                tasa_ultima_mejora = tasa_actual
                vecinos_consecutivos = 0
                break

    return pesos


# FUNCIONES NECESARIAS PARA ALGORITMO RELIEF

def enemigo_mas_cercano(dato, clase, datos, clases):
    indices_enemigos = np.where(clases != clase)[0]
    enemigos = datos[indices_enemigos]

    distancias = (enemigos - dato)**2
    distancias = np.sum(distancias, axis=1)
    distancias = np.sqrt(distancias)

    return enemigos[np.argmin(distancias)]

def amigo_mas_cercano(dato, clase, datos, clases):
    indices_amigos = np.where(clases == clase)[0]
    amigos = datos[indices_amigos]

    distancias = (amigos-dato)**2
    distancias = np.sum(distancias, axis=1)
    distancias = np.sqrt(distancias)
    distancias[distancias == 0] = np.inf

    return amigos[np.argmin(distancias)]

def relief(datos, clases):
    w = np.zeros(datos.shape[1])

    for i, ei in enumerate(datos):
        ee = enemigo_mas_cercano(ei, clases[i], datos, clases)
        ea = amigo_mas_cercano(ei, clases[i], datos, clases)
        w = w + np.abs(ei - ee) - np.abs(ei - ea)

    w = np.maximum(0, w)
    w = w / np.max(w)

    return w


if __name__ == '__main__':
    if len(sys.argv) == 7:
        # LEEMOS LOS 5 FICHEROS
        nombres1, datos1, clases1 = faux.leer_datos(sys.argv[1])
        nombres2, datos2, clases2 = faux.leer_datos(sys.argv[2])
        nombres3, datos3, clases3 = faux.leer_datos(sys.argv[3])
        nombres4, datos4, clases4 = faux.leer_datos(sys.argv[4])
        nombres5, datos5, clases5 = faux.leer_datos(sys.argv[5])
        datos_todos = datos1 + datos2 + datos3 + datos4 + datos5
        clases_todos = clases1 + clases2 + clases3 + clases4 + clases5

        # CONVERTIMOS LAS LISTAS DE STRING NUMERICOS A ARRAYS DE NUMPY Y NORMALIZAMOS
        datos_todos = np.array(datos_todos, dtype=float)
        datos_todos = faux.normalizar_datos(datos_todos)

        # DIVIDIMOS LOS DATOS YA NORMALIZADOS
        datos1 = datos_todos[:len(datos1)]
        datos2 = datos_todos[len(datos1):len(datos1)+len(datos2)]
        datos3 = datos_todos[len(datos1)+len(datos2):len(datos1)+len(datos2)+len(datos3)]
        datos4 = datos_todos[len(datos1)+len(datos2)+len(datos3):len(datos1)+len(datos2)+len(datos3)+len(datos4)]
        datos5 = datos_todos[len(datos1)+len(datos2)+len(datos3)+len(datos4):]

        # COMO DEPENDIENDO DEL FICHERO, LAS CLASES CONTIENEN UN STRING O UN VALOR
        # NUMÉRICO, CREAMOS UN DICCIONARIO PARA QUEDARNOS CON CLASES NUMÉRICAS
        unicos = np.unique(clases_todos)
        diccionario = {}
        for i, s in enumerate(unicos):
            diccionario[s] = i

        print('')
        print('EQUIVALENCIAS DE CLASES:')
        for clave, valor in diccionario.items():
            print(clave, 'equivale a la clase', valor)

        clases_todos = np.array(np.vectorize(diccionario.get)(clases_todos))

        # DIVIDIMOS LAS CLASES
        clases1 = clases_todos[:len(clases1)]
        clases2 = clases_todos[len(clases1):len(clases1) + len(clases2)]
        clases3 = clases_todos[len(clases1) + len(clases2):len(clases1) + len(clases2) + len(clases3)]
        clases4 = clases_todos[len(clases1) + len(clases2) + len(clases3):len(clases1) + len(clases2) + len(clases3) + len(clases4)]
        clases5 = clases_todos[len(clases1) + len(clases2) + len(clases3) + len(clases4):]

        # PARAMETRO FITNESS
        alpha = 0.8

        # INICIALIZAMOS SEMILLA
        np.random.seed(int(sys.argv[6]))

        # PARA CONSTRUIR LA TABLA
        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        datos_todos = [datos1,datos2,datos3,datos4,datos5]
        clases_todos = [clases1,clases2,clases3,clases4,clases5]

        # ALGORITMO 1-NN
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO 1NN')
        for i in range(len(datos_todos)):
            # pesos de 1nn
            pesos = np.ones(datos_todos[0].shape[1])
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas = faux.tasa_clas(pesos, datos_test, clases_test, datos_train, clases_train)
            tred = faux.tasa_red(pesos)
            tfit = alpha * tclas + (1 - alpha) * tred
            fin = timeit.default_timer()
            tiempo = fin - inicio
            # añadimos los resultados a la tabla
            tabla.append(['Particion ' + str(i + 1), tclas, tred, tfit, '{:0.5f}'.format(tiempo)])
            # calculo media y std
            tclas_medio = tclas_medio + tclas
            tclas_std = tclas_std + tclas**2
            tred_medio = tred_medio + tred
            tred_std = tred_std + tred**2
            tfit_medio = tfit_medio + tfit
            tfit_std = tfit_std + tfit**2
            tiempo_medio = tiempo_medio + tiempo
            tiempo_std = tiempo_std + tiempo**2

        # calculo media y std
        tclas_medio = tclas_medio / len(datos_todos)
        tred_medio = tred_medio / len(datos_todos)
        tfit_medio = tfit_medio / len(datos_todos)
        tiempo_medio = tiempo_medio / len(datos_todos)
        tclas_std = np.sqrt(tclas_std / len(datos_todos) - tclas_medio**2)
        tred_std = np.sqrt(tred_std / len(datos_todos) - tred_medio**2)
        tfit_std = np.sqrt(tfit_std / len(datos_todos) - tfit_medio**2)
        tiempo_std = np.sqrt(tiempo_std / len(datos_todos) - tiempo_medio**2)
        # añadimos los datos a la tabla
        tabla.append(['Media', tclas_medio, tred_medio, tfit_medio, tiempo_medio])
        tabla.append(['Std', tclas_std, tred_std, tfit_std, tiempo_std])
        # imprimimos la tabla
        print(tabulate(tabla,headers='firstrow'))

        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        # ALGORITMO RELIEF
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO RELIEF')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # pesos + clasificacion + tiempo
            inicio = timeit.default_timer()
            pesos = relief(datos_train, clases_train)
            tclas = faux.tasa_clas(pesos, datos_test, clases_test, datos_train, clases_train)
            tred = faux.tasa_red(pesos)
            tfit = alpha * tclas + (1 - alpha) * tred
            fin = timeit.default_timer()
            tiempo = fin-inicio
            # mostramos pesos
            print('PESOS PARTICION', i + 1, ':', pesos)
            # añadimos resultados a la tabla
            tabla.append(['Particion ' + str(i + 1), tclas, tred, tfit, '{:0.5f}'.format(tiempo)])
            # calculo media y std
            tclas_medio = tclas_medio + tclas
            tclas_std = tclas_std + tclas**2
            tred_medio = tred_medio + tred
            tred_std = tred_std + tred**2
            tfit_medio = tfit_medio + tfit
            tfit_std = tfit_std + tfit**2
            tiempo_medio = tiempo_medio + tiempo
            tiempo_std = tiempo_std + tiempo**2

        # calculo media y std
        tclas_medio = tclas_medio / len(datos_todos)
        tred_medio = tred_medio / len(datos_todos)
        tfit_medio = tfit_medio / len(datos_todos)
        tiempo_medio = tiempo_medio / len(datos_todos)
        tclas_std = np.sqrt(tclas_std / len(datos_todos) - tclas_medio**2)
        tred_std = np.sqrt(tred_std / len(datos_todos) - tred_medio**2)
        tfit_std = np.sqrt(tfit_std / len(datos_todos) - tfit_medio**2)
        tiempo_std = np.sqrt(tiempo_std / len(datos_todos) - tiempo_medio**2)
        # añadimos los datos a la tabla
        tabla.append(['Media', tclas_medio, tred_medio, tfit_medio, tiempo_medio])
        tabla.append(['Std', tclas_std, tred_std, tfit_std, tiempo_std])
        # imprimimos la tabla
        print(tabulate(tabla,headers='firstrow'))

        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        # ALGORITMO BUSQUEDA LOCAL
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO BUSQUEDA LOCAL')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            n_max = 200*datos_train.shape[1]
            # pesos + clasificacion + tiempo
            inicio = timeit.default_timer()
            pesos = busqueda_local(n_max_vecinos=n_max, n_max_evaluaciones=15000, delta=0.3, datos=datos_train, clases=clases_train)
            tclas = faux.tasa_clas(pesos, datos_test, clases_test, datos_train, clases_train)
            tred = faux.tasa_red(pesos)
            tfit = alpha * tclas + (1 - alpha) * tred
            fin = timeit.default_timer()
            tiempo = fin - inicio
            # mostramos pesos
            print('PESOS PARTICION', i + 1, ':', pesos)
            # añadimos resultados a la tabla
            tabla.append(['Particion ' + str(i + 1), tclas, tred, tfit, '{:0.5f}'.format(tiempo)])
            # calculo media y std
            tclas_medio = tclas_medio + tclas
            tclas_std = tclas_std + tclas**2
            tred_medio = tred_medio + tred
            tred_std = tred_std + tred**2
            tfit_medio = tfit_medio + tfit
            tfit_std = tfit_std + tfit**2
            tiempo_medio = tiempo_medio + tiempo
            tiempo_std = tiempo_std + tiempo**2

        # calculo media y std
        tclas_medio = tclas_medio / len(datos_todos)
        tred_medio = tred_medio / len(datos_todos)
        tfit_medio = tfit_medio / len(datos_todos)
        tiempo_medio = tiempo_medio / len(datos_todos)
        tclas_std = np.sqrt(tclas_std / len(datos_todos) - tclas_medio**2)
        tred_std = np.sqrt(tred_std / len(datos_todos) - tred_medio**2)
        tfit_std = np.sqrt(tfit_std / len(datos_todos) - tfit_medio**2)
        tiempo_std = np.sqrt(tiempo_std / len(datos_todos) - tiempo_medio**2)
        # añadimos los datos a la tabla
        tabla.append(['Media', tclas_medio, tred_medio, tfit_medio, tiempo_medio])
        tabla.append(['Std', tclas_std, tred_std, tfit_std, tiempo_std])
        # imprimimos la tabla
        print(tabulate(tabla, headers='firstrow'))

    else:
        print("ERROR: Número incorrecto de parámetros.")
        print("USO: <interprete-python> <ejecutable> <fichero-1> <fichero-2> <fichero-3> <fichero-4> <fichero-5> <semilla>")