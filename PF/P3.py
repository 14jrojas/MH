import sys
import numpy as np
import timeit
from tabulate import tabulate

import faux

# HIPERPARÁMETROS
ALPHA_FITNESS = 0.8
SIGMA_MOV = 0.3
MAX_ITERS_ES = 15000
ITERS_BMB = 15
MAX_EVALUACIONES_BMB = 1000
KMAX_VNS = 3

# Reciclado P2
def mov(pesos, indice, delta):
    pesos[indice] = pesos[indice] + np.random.normal(loc = 0, scale = delta)
    if pesos[indice] > 1: pesos[indice] = 1
    if pesos[indice] < 0: pesos[indice] = 0
    return pesos

# Reciclado P1
def busqueda_local(n_max_vecinos, n_max_evaluaciones, delta, datos, clases, pesos = []):
    if len(pesos) == 0:
        pesos = np.random.uniform(0, 1, datos.shape[1])

    indices = np.arange(len(pesos))

    tasa_ultima_mejora = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos, clases, datos, clases)

    vecinos_consecutivos = 0
    evaluaciones = 0
    while(vecinos_consecutivos < n_max_vecinos and evaluaciones < n_max_evaluaciones):
        np.random.shuffle(indices)
        for indice in indices:
            copia = np.copy(pesos)
            copia = mov(copia, indice, delta=SIGMA_MOV)
            vecinos_consecutivos = vecinos_consecutivos + 1
            if vecinos_consecutivos >= n_max_vecinos: break
            tasa_actual = faux.tasa_fitness(ALPHA_FITNESS, copia, datos, clases, datos, clases)
            evaluaciones = evaluaciones + 1
            if (tasa_actual > tasa_ultima_mejora):
                pesos = copia
                tasa_ultima_mejora = tasa_actual
                vecinos_consecutivos = 0
                break

    return pesos

# ALGORITMO ENFRIAMIENTO SIMULADO
def ES(datos_train, clases_train, datos_test, clases_test, pesos = [], max_iters=MAX_ITERS_ES):
    # inicializamos el vector de pesos
    if len(pesos) == 0:
        pesos = np.random.uniform(0, 1, datos_train.shape[1])
    
    # calculamos sus valores iniciales
    fitness_actual = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos_train, clases_train, datos_train, clases_train)
    mejor_fitness = fitness_actual
    mejor_pesos = pesos.copy()

    # inicializamos parametros del bucle
    max_vecinos = 10 * len(pesos)
    max_exitos = 0.1 * max_vecinos
    # numero de enfriamientos
    M = round(max_iters / max_vecinos)

    # inicializamos temperaturas
    T0 = (0.3 * fitness_actual) / (-np.log(0.2))
    Tf = 0.0001

    # esto no se por que lo hace con ese factor
    # imponemos que Tf sea menor que T0
    while Tf > T0:
        Tf *= 0.0001

    # esquema de Cauchy modificado
    beta = (T0 - Tf) / (M * T0 * Tf)

    # valores iniciales
    T_actual = T0
    n_exitos = 1
    it = 1

    # bucle algoritmo
    while (n_exitos != 0 and it <= max_iters and T_actual > Tf):
        n_exitos = 0
        n_vecinos = 0

        while(n_exitos < max_exitos and n_vecinos < max_vecinos):
            # mutamos
            w = pesos.copy()
            i = np.random.randint(0, len(w)-1)
            w = mov(w, i, SIGMA_MOV)
            nuevo_fitness = faux.tasa_fitness(ALPHA_FITNESS, w, datos_train, clases_train, datos_train, clases_train)
            it = it + 1
            
            diferencia = nuevo_fitness - fitness_actual
            # si nuevo fitness es mejor o aceptamos con una probabilidad
            if diferencia > 0 or np.random.uniform(0,1) <= np.exp(diferencia / T_actual):
                fitness_actual = nuevo_fitness
                pesos = w.copy()
                n_exitos = n_exitos + 1
                # si supera al mejor actualizamos
                if fitness_actual > mejor_fitness:
                    mejor_fitness = fitness_actual
                    mejor_pesos = pesos
                
            n_vecinos = n_vecinos + 1

        # actualizacion T
        T_actual = T_actual / (1 + beta*T_actual)

    # estamos usando ES
    if max_iters == MAX_ITERS_ES:
        # resultados
        tclas = faux.tasa_clas(mejor_pesos, datos_test, clases_test, datos_train, clases_train)
        tred = faux.tasa_red(mejor_pesos)
        tfit = ALPHA_FITNESS*tclas + (1-ALPHA_FITNESS)*tred

        return tclas, tred, tfit
    # estamos usando ES-ILS
    else:
        return mejor_pesos


# ALGORITMO BUSQUEDA MULTIARRANQUE BASICA
def BMB(datos_train, clases_train, datos_test, clases_test):
    # inicializamos pesos
    pesos_actual = np.random.uniform(0, 1, datos_train.shape[1])
    mejor_fitness = faux.tasa_fitness(ALPHA_FITNESS, pesos_actual, datos_train, clases_train, datos_train, clases_train)
    mejor_pesos = pesos_actual.copy()

    for i in range(ITERS_BMB):
        # busqueda local + actualizar valores
        pesos_actual = busqueda_local(len(mejor_pesos), MAX_EVALUACIONES_BMB, SIGMA_MOV, datos_train, clases_train, pesos_actual)
        fitness_actual = faux.tasa_fitness(ALPHA_FITNESS, pesos_actual, datos_train, clases_train, datos_train, clases_train)

        # si hay mejora, actualizamos
        if fitness_actual > mejor_fitness:
            mejor_pesos = np.copy(pesos_actual)
            mejor_fitness = fitness_actual

        # inicializamos pesos para siguiente iteracion
        pesos_actual = np.random.uniform(0, 1, datos_train.shape[1])

    # resultados
    tclas = faux.tasa_clas(mejor_pesos, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(mejor_pesos)
    tfit = ALPHA_FITNESS*tclas + (1-ALPHA_FITNESS)*tred

    return tclas, tred, tfit

# ALGORITMO BUSQUEDA LOCAL REITERADA
def ILS(datos_train, clases_train, datos_test, clases_test):
    # inicializamos pesos
    pesos = np.random.uniform(0, 1, datos_train.shape[1])

    # busqueda local a la solucion inicial
    pesos = busqueda_local(len(pesos), MAX_EVALUACIONES_BMB, SIGMA_MOV, datos_train, clases_train, pesos)
    mejor_fitness = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos_train, clases_train, datos_train, clases_train)
    mejor_pesos = pesos.copy()

    # numero de mutaciones bruscas
    mutaciones = round(0.1 * len(pesos))
    if mutaciones < 2: mutaciones = 2

    for i in range(ITERS_BMB-1):
        # mutacion brusca
        for _ in range(mutaciones):
            indice = np.random.randint(0, len(pesos)-1)
            pesos[indice] = np.random.uniform(0, 1)

        # busqueda local a pesos actuales
        pesos = busqueda_local(len(pesos), MAX_EVALUACIONES_BMB, SIGMA_MOV, datos_train, clases_train, pesos)
        fitness_actual = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos_train, clases_train, datos_train, clases_train)
        
        # si hay mejora, actualizamos
        if fitness_actual > mejor_fitness:
            mejor_fitness = fitness_actual
            mejor_pesos = pesos
        
        # repetimos sobre los mejores pesos
        pesos = mejor_pesos.copy()

    # resultados
    tclas = faux.tasa_clas(mejor_pesos, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(mejor_pesos)
    tfit = ALPHA_FITNESS*tclas + (1-ALPHA_FITNESS)*tred

    return tclas, tred, tfit

# ALGORITMO BUSQUEDA LOCAL REITERADA CON ENFRIAMIENTO SIMULADO
def ILS_ES(datos_train, clases_train, datos_test, clases_test):
    # inicializamos pesos
    pesos = np.random.uniform(0, 1, datos_train.shape[1])

    # ES a la solucion inicial
    pesos = ES(datos_train, clases_train, datos_test, clases_test, pesos, MAX_EVALUACIONES_BMB)
    mejor_fitness = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos_train, clases_train, datos_train, clases_train)
    mejor_pesos = pesos.copy()

    # numero de mutaciones bruscas
    mutaciones = round(0.1 * len(pesos))
    if mutaciones < 2: mutaciones = 2

    for i in range(ITERS_BMB-1):
        # mutacion brusca
        for _ in range(mutaciones):
            indice = np.random.randint(0, len(pesos)-1)
            pesos[indice] = np.random.uniform(0, 1)

        # ES a pesos actuales
        pesos = ES(datos_train, clases_train, datos_test, clases_test, pesos, MAX_EVALUACIONES_BMB)
        fitness_actual = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos_train, clases_train, datos_train, clases_train)
        
        # si hay mejora, actualizamos
        if fitness_actual > mejor_fitness:
            mejor_fitness = fitness_actual
            mejor_pesos = pesos
        
        # repetimos sobre los mejores pesos
        pesos = mejor_pesos.copy()

    # resultados
    tclas = faux.tasa_clas(mejor_pesos, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(mejor_pesos)
    tfit = ALPHA_FITNESS*tclas + (1-ALPHA_FITNESS)*tred

    return tclas, tred, tfit

# ALGORITMO BUSQUEDA DE VECINDARIO VARIABLE
def busqueda_local_modificada(n_max_vecinos, n_max_evaluaciones, delta, datos, clases, pesos=[], k=KMAX_VNS):
    if len(pesos) == 0:
        pesos = np.random.uniform(0, 1, datos.shape[1])

    tasa_ultima_mejora = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos, clases, datos, clases)

    vecinos_consecutivos = 0
    evaluaciones = 0
    while(vecinos_consecutivos < n_max_vecinos and evaluaciones < n_max_evaluaciones):
        indice = np.random.randint(0, len(pesos)-1)
        copia = np.copy(pesos)
        copia = mov(copia, indice, delta=SIGMA_MOV)
        vecinos_consecutivos = vecinos_consecutivos + 1
        for i in range(k-1):
            indice = np.random.randint(0, len(pesos)-1)
            copia = mov(copia, indice, delta=SIGMA_MOV)
        tasa_actual = faux.tasa_fitness(ALPHA_FITNESS, copia, datos, clases, datos, clases)
        evaluaciones = evaluaciones + 1
        if (tasa_actual > tasa_ultima_mejora):
            pesos = copia
            tasa_ultima_mejora = tasa_actual
            vecinos_consecutivos = 0

    return pesos

def VNS(datos_train, clases_train, datos_test, clases_test, k_max=KMAX_VNS):
    # inicializamos pesos
    pesos = np.random.uniform(0, 1, datos_train.shape[1])
    k = 0

    # busqueda local a la solucion inicial
    pesos = busqueda_local_modificada(len(pesos), MAX_EVALUACIONES_BMB, SIGMA_MOV, datos_train, clases_train, pesos=pesos, k=(k % k_max)+1)
    mejor_fitness = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos_train, clases_train, datos_train, clases_train)
    mejor_pesos = pesos.copy()

    # numero de mutaciones bruscas
    mutaciones = round(0.1 * len(pesos))
    if mutaciones < 2: mutaciones = 2

    for i in range(ITERS_BMB-1):
        # mutacion brusca
        for _ in range(mutaciones):
            indice = np.random.randint(0, len(pesos)-1)
            pesos[indice] = np.random.uniform(0, 1)

        # busqueda local a pesos actuales
        pesos = busqueda_local_modificada(len(pesos), MAX_EVALUACIONES_BMB, SIGMA_MOV, datos_train, clases_train, pesos=pesos, k=(k % k_max)+1)
        fitness_actual = faux.tasa_fitness(ALPHA_FITNESS, pesos, datos_train, clases_train, datos_train, clases_train)
        
        # si hay mejora, actualizamos
        if fitness_actual > mejor_fitness:
            mejor_fitness = fitness_actual
            mejor_pesos = pesos
            k = 0
        else:
            k = k+1
        
        # repetimos sobre los mejores pesos
        pesos = mejor_pesos.copy()

    # resultados
    tclas = faux.tasa_clas(mejor_pesos, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(mejor_pesos)
    tfit = ALPHA_FITNESS*tclas + (1-ALPHA_FITNESS)*tred

    return tclas, tred, tfit

# ____________________________________________________________________________

# MAIN
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
        alpha = ALPHA_FITNESS

        # INICIALIZAMOS SEMILLA
        np.random.seed(int(sys.argv[6]))

        # PARA CONSTRUIR LA TABLA
        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        datos_todos = [datos1,datos2,datos3,datos4,datos5]
        clases_todos = [clases1,clases2,clases3,clases4,clases5]

        # ALGORITMO ES
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO ES')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = ES(datos_train, clases_train, datos_test, clases_test, pesos = [], max_iters=MAX_ITERS_ES)
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

        # ALGORITMO BMB
        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO BMB')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = BMB(datos_train, clases_train, datos_test, clases_test)
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
        

        # ALGORITMO ILS
        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO ILS')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = ILS(datos_train, clases_train, datos_test, clases_test)
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


        # ALGORITMO ILS-ES
        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO ILS-ES')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = ILS_ES(datos_train, clases_train, datos_test, clases_test)
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


        # ALGORITMO VNS
        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO VNS')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = VNS(datos_train, clases_train, datos_test, clases_test, k_max=KMAX_VNS)
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

    else:
        print("ERROR: Número incorrecto de parámetros.")
        print("USO: <interprete-python> <ejecutable> <fichero-1> <fichero-2> <fichero-3> <fichero-4> <fichero-5> <semilla>")