import sys
import numpy as np
import timeit
import random
import copy
from tabulate import tabulate

import faux

# HIPERPARÁMETROS
# generico
ALPHA_FITNESS = 0.8
N_MAX_EVALUACIONES = 15000
# coa
COYOTES_POR_PACK = 5
N_PACKS = 10
# memeticos
PLS = 0.1
FRECUENCIA_MEMETICO = 20
DELTA_BL = 0.3
# mejora
R_CAOS = 4
X0 = [np.random.rand()]
PCAMBIO = 0.3

# clase coyote
class coyote:
    def __init__(self, datos, clases, pesos=[], edad=0, caotico=False):
        if len(pesos) == 0:
            if not caotico:
                self.pesos = np.random.uniform(0,1,len(datos[0]))
            else:
                self.pesos = np.array([siguiente_numero_caotico(X0)[0] for _ in range(len(datos[0]))])
        else:
            self.pesos = pesos.copy()

        self.tclas = faux.tasa_clas(self.pesos, datos, clases, datos, clases)
        self.tred = faux.tasa_red(self.pesos)
        self.tfit = ALPHA_FITNESS * self.tclas + (1-ALPHA_FITNESS) * self.tred
        self.edad = edad

# funcion que inicializa la poblacion de coyotes
def inicializar_coyotes(datos_train, clases_train, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK):
    poblacion = []
    for i in range(npacks):
        paquete = []
        for j in range(coyotespp):
            c = coyote(datos_train, clases_train)
            paquete.append(c)
        poblacion.append(paquete)

    return poblacion

# funcion que calcula la tendencia de un paquete
def calcular_tendencia_cultural(pack):
    pesos_pack = np.array([coyote.pesos for coyote in pack])
    medianas = np.median(pesos_pack, axis=0)
    return medianas

# funcion que crea un nuevo coyote
def nacimiento(pack, datos_train, clases_train, pa, ps):
    # pesos del coyote a nacer
    pesos_puppy = np.random.uniform(0,1,len(datos_train[0]))

    # elegimos dos padres aleatorios
    rc1 = np.random.randint(COYOTES_POR_PACK)
    rc2 = rc1
    while rc2 == rc1:
        rc2 = np.random.randint(COYOTES_POR_PACK)

    # indices de las caracteristicas desordenados
    indices = np.random.permutation(len(datos_train[0]))

    # elegimos una caracteristica de cada padre a conservar
    d1 = indices.item(0)
    indices = np.delete(indices, 0)
    d2 = indices.item(0)
    indices = np.delete(indices, 0)

    # conservamos dicha raracteristica
    pesos_puppy[d1] = pack[rc1].pesos[d1]
    pesos_puppy[d2] = pack[rc2].pesos[d2]

    # del resto, conservamos la de rc1 con probabilidad pa y la de rc2 con probabilidad 1-pa
    # si no se da ninguna condicion, inicialización aleatoria
    for i in indices:
        prob = np.random.rand()
        if prob < ps:
            pesos_puppy[i] = pack[rc1].pesos[i]
        else:
            if prob >= ps+pa:
                pesos_puppy[i] = pack[rc2].pesos[i]
            # else
            # la inicializacion aleatoria se hace arriba
    
    # nace el coyote
    puppy = coyote(datos_train, clases_train, pesos_puppy)

    return puppy

# COYOTE OPTIMIZATION ALGORITHM + COYOTE OPTIMIZATION ALGORITHM MEJORADO
def COA(datos_train, clases_train, datos_test, clases_test, caos=False, cambio_alfa=False, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES):
    # inicializamos la poblacion
    poblacion = []
    if caos:
        poblacion = inicializar_coyotes_caotico(datos_train, clases_train, npacks, coyotespp)
    else:
        poblacion = inicializar_coyotes(datos_train, clases_train, npacks, coyotespp)
    # actualizamos el numero de evaluaciones
    evaluaciones = 0
    evaluaciones += npacks * coyotespp

    # probabilidad de intercambio
    pi = 0.005 * coyotespp**2
    # probabilidad de dispersion
    ps = 1 / len(datos_train[0])
    # probabilidad de asociacion
    pa = (1-ps) / 2

    # bucle principal
    while evaluaciones < max_evaluaciones:
        for pack in poblacion:
            # escogemos el coyote alfa
            if cambio_alfa:
                if np.random.rand() < PCAMBIO:
                    alfa = min(pack, key=lambda x: x.edad)
                else:
                    alfa = max(pack, key=lambda x: x.tfit)
            else:
                alfa = max(pack, key=lambda x: x.tfit)

            # calculamos la tendencia
            tendencia_cultural = calcular_tendencia_cultural(pack)

            # intercambio cultural
            for c in range(coyotespp):
                pesos_actual = pack[c].pesos
                # elegimos dos coyotes distintos
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(coyotespp)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(coyotespp)

                # actualizamos la condicion social según alfa y tendencia
                if caos:
                    r1 = siguiente_numero_caotico(X0)[0]
                    r2 = siguiente_numero_caotico(X0)[0]
                else:
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                delta1 = alfa.pesos - pack[rc1].pesos
                delta2 = tendencia_cultural - pack[rc2].pesos
                pesos_actual = pesos_actual + r1*delta1 + r2*delta2
                pesos_actual = np.clip(pesos_actual, 0, 1)

                # creamos el nuevo coyote con los pesos
                nuevo_c = coyote(datos_train, clases_train, pesos_actual)
                evaluaciones += 1
                # si es mejor, sustituimos
                if nuevo_c.tfit > pack[c].tfit:
                    pack[c] = nuevo_c
            
            # nuevo cachorro
            if caos:
                puppy = nacimiento_caotico(pack, datos_train, clases_train, pa, ps)
            else:
                puppy = nacimiento(pack, datos_train, clases_train, pa, ps)
            evaluaciones += 1

            # vemos si puppy supera en fitness
            pack = sorted(pack, key=lambda x: (x.tfit, -x.edad))
            if puppy.tfit > pack[0].tfit:
                pack[0] = puppy
            random.shuffle(pack)

        # intercambio de manada
        if np.random.rand() < pi:
            p1 = np.random.randint(npacks)
            p2 = np.random.randint(npacks)
            c1 = np.random.randint(coyotespp)
            c2 = np.random.randint(coyotespp)

            aux = poblacion[p1][c1]
            poblacion[p1][c1] = poblacion[p2][c2]
            poblacion[p2][c2] = aux

        # aumentamos la edad
        for pack in poblacion:
            for elem in pack:
                elem.edad += 1

    # calculamos el mejor alfa de los alfas
    alfa = max((max(pack, key=lambda x: x.tfit) for pack in poblacion), key=lambda x: x.tfit)

    # resultados
    w_final = alfa.pesos
    tclas = faux.tasa_clas(w_final, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(w_final)
    tfit = ALPHA_FITNESS*tclas + (1-ALPHA_FITNESS)*tred

    return tclas, tred, tfit

# Modificado P2
def mov(pesos, indice, delta):
    pesos[indice] = pesos[indice] + np.random.normal(loc = 0, scale = delta)
    if pesos[indice] > 1: pesos[indice] = 1
    if pesos[indice] < 0: pesos[indice] = 0
    return pesos

# BUSQUEDA LOCAL BAJA INTENSIDAD
def busqueda_local_bi(datos_train, clases_train, coyote):
    mejor_tfit = coyote.tfit
    n = len(coyote.pesos)
    indices = np.arange(len(coyote.pesos))
    
    it = 0
    while it < 8*n:
        np.random.shuffle(indices)
        for indice in indices:
            it = it + 1
            if it >= 8*n: break
            copia = copy.deepcopy(coyote)
            copia.pesos = mov(copia.pesos, indice, delta=DELTA_BL)
            copia.tred = faux.tasa_red(copia.pesos)
            copia.tclas = faux.tasa_clas(copia.pesos, datos_train, clases_train, datos_train, clases_train)
            copia.tfit = ALPHA_FITNESS * copia.tclas + (1-ALPHA_FITNESS) * copia.tred
            if (copia.tfit > mejor_tfit):
                coyote = copia
                mejor_tfit = copia.tfit
                break

    return it, coyote

# ALGORITMO MEMETICO A SOLUCION ALEATORIA CON P=0.1
def AM_Rand(poblacion, indices, datos_train, clases_train):
    it = 0
    random.shuffle(indices)
    for i in range(int(PLS*len(indices))):
        j,poblacion[indices[i][0]][indices[i][1]] = busqueda_local_bi(datos_train, clases_train, poblacion[indices[i][0]][indices[i][1]])
        it = it + j
    
    return it, poblacion

# ALGORITMO MEMETICO A 0.1*N MEJORES
def AM_Best(poblacion, indices, datos_train, clases_train):
    it = 0
    indices = sorted(indices, key=lambda x: poblacion[x[0]][x[1]].tfit , reverse=True)
    for i in range(int(PLS*len(indices))):
        j,poblacion[indices[i][0]][indices[i][1]] = busqueda_local_bi(datos_train, clases_train, poblacion[indices[i][0]][indices[i][1]])
        it = it + j
    
    return it, poblacion

# COYOTE OPTIMIZATION ALGORITHM + MEMETICO
def COA_memetico(datos_train, clases_train, datos_test, clases_test, tipo_memetico, caos=False, cambio_alfa=False, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES):
    # inicializamos la poblacion
    poblacion = []
    if caos:
        poblacion = inicializar_coyotes_caotico(datos_train, clases_train, npacks, coyotespp)
    else:
        poblacion = inicializar_coyotes(datos_train, clases_train, npacks, coyotespp)
    # actualizamos el numero de evaluaciones
    evaluaciones = 0
    evaluaciones += npacks * coyotespp
    # añadido memetico
    generacion = 0

    # probabilidad de intercambio
    pi = 0.005 * coyotespp**2
    # probabilidad de dispersion
    ps = 1 / len(datos_train[0])
    # probabilidad de asociacion
    pa = (1-ps) / len(datos_train[0])

    # bucle principal
    while evaluaciones < max_evaluaciones:
        for pack in poblacion:
            # escogemos el coyote alfa
            if cambio_alfa:
                if np.random.rand() < PCAMBIO:
                    alfa = min(pack, key=lambda x: x.edad)
                else:
                    alfa = max(pack, key=lambda x: x.tfit)
            else:
                alfa = max(pack, key=lambda x: x.tfit)

            # calculamos la tendencia
            tendencia_cultural = calcular_tendencia_cultural(pack)

            # intercambio cultural
            for c in range(coyotespp):
                pesos_actual = pack[c].pesos
                # elegimos dos coyotes distintos
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(coyotespp)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(coyotespp)

                # actualizamos la condicion social según alfa y tendencia
                if caos:
                    r1 = siguiente_numero_caotico(X0)[0]
                    r2 = siguiente_numero_caotico(X0)[0]
                else:
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                delta1 = alfa.pesos - pack[rc1].pesos
                delta2 = tendencia_cultural - pack[rc2].pesos
                pesos_actual = pesos_actual + r1*delta1 + r2*delta2
                pesos_actual = np.clip(pesos_actual, 0, 1)

                # creamos el nuevo coyote con los pesos
                nuevo_c = coyote(datos_train, clases_train, pesos_actual)
                evaluaciones += 1
                # si es mejor, sustituimos
                if nuevo_c.tfit > pack[c].tfit:
                    pack[c] = nuevo_c
            
            # nuevo cachorro
            if caos:
                puppy = nacimiento_caotico(pack, datos_train, clases_train, pa, ps)
            else:
                puppy = nacimiento(pack, datos_train, clases_train, pa, ps)
            evaluaciones += 1

            # vemos si puppy supera en fitness
            pack = sorted(pack, key=lambda x: (x.tfit, -x.edad))
            if puppy.tfit > pack[0].tfit:
                pack[0] = puppy
            random.shuffle(pack)

        # intercambio de manada
        if np.random.rand() < pi:
            p1 = np.random.randint(npacks)
            p2 = np.random.randint(npacks)
            c1 = np.random.randint(coyotespp)
            c2 = np.random.randint(coyotespp)

            aux = poblacion[p1][c1]
            poblacion[p1][c1] = poblacion[p2][c2]
            poblacion[p2][c2] = aux

        # aumentamos la edad
        for pack in poblacion:
            for elem in pack:
                elem.edad += 1

        generacion += 1
        # aplicamos busqueda local a los alfas
        if generacion % FRECUENCIA_MEMETICO == 0:
            indices_alfas = [(j, max(range(len(pack)), key=lambda x: pack[x].tfit)) for j,pack in enumerate(poblacion)]
            j,poblacion = tipo_memetico(poblacion, indices_alfas, datos_train, clases_train)
            evaluaciones += j


    # calculamos el mejor alfa de los alfas
    alfa = max((max(pack, key=lambda x: x.tfit) for pack in poblacion), key=lambda x: x.tfit)

    # resultados
    w_final = alfa.pesos
    tclas = faux.tasa_clas(w_final, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(w_final)
    tfit = ALPHA_FITNESS*tclas + (1-ALPHA_FITNESS)*tred

    return tclas, tred, tfit

# funcion que genera un número caótico
def siguiente_numero_caotico(x):
    x[0] = R_CAOS * x[0] * (1-x[0])
    return x

# funcion que inicializa la poblacion con números caóticos
def inicializar_coyotes_caotico(datos_train, clases_train, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK):
    poblacion = []
    for i in range(npacks):
        paquete = []
        for j in range(coyotespp):
            c = coyote(datos_train, clases_train, caotico=True)
            paquete.append(c)
        poblacion.append(paquete)

    return poblacion

# funcion que crea un nuevo coyote con números caóticos
def nacimiento_caotico(pack, datos_train, clases_train, pa, ps):
    # pesos del coyote a nacer
    pesos_puppy = np.array([siguiente_numero_caotico(X0)[0] for _ in range(len(datos_train[0]))])

    # elegimos dos padres aleatorios
    rc1 = np.random.randint(COYOTES_POR_PACK)
    rc2 = rc1
    while rc2 == rc1:
        rc2 = np.random.randint(COYOTES_POR_PACK)

    # indices de las caracteristicas desordenados
    indices = np.random.permutation(len(datos_train[0]))

    # elegimos una caracteristica de cada padre a conservar
    d1 = indices.item(0)
    indices = np.delete(indices, 0)
    d2 = indices.item(0)
    indices = np.delete(indices, 0)

    # conservamos dicha raracteristica
    pesos_puppy[d1] = pack[rc1].pesos[d1]
    pesos_puppy[d2] = pack[rc2].pesos[d2]

    # del resto, conservamos la de rc1 con probabilidad pa y la de rc2 con probabilidad 1-pa
    # si no se da ninguna condicion, inicialización aleatoria
    for i in indices:
        prob = np.random.rand()
        if prob < ps:
            pesos_puppy[i] = pack[rc1].pesos[i]
        else:
            if prob >= ps+pa:
                pesos_puppy[i] = pack[rc2].pesos[i]
            # else
            # la inicializacion caotica se hace arriba
    
    # nace el coyote
    puppy = coyote(datos_train, clases_train, pesos_puppy)

    return puppy


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

        # ALGORITMO COA
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO COA')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = COA(datos_train, clases_train, datos_test, clases_test, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES)
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


        # ALGORITMO COA_Memetico (Rand)
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
        print('ALGORITMO COA_Memetico (Rand)')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = COA_memetico(datos_train, clases_train, datos_test, clases_test, AM_Rand, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES)
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


        # ALGORITMO COA_Memetico (Best)
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
        print('ALGORITMO COA_Memetico (Best)')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = COA_memetico(datos_train, clases_train, datos_test, clases_test, AM_Best, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES)
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

        # ALGORITMO COA_Caotico
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
        print('ALGORITMO COA_Caotico')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = COA(datos_train, clases_train, datos_test, clases_test, caos=True, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES)
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

        
        # ALGORITMO COA_Cambio-Alfa
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
        print('ALGORITMO COA_Cambio-Alfa')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = COA(datos_train, clases_train, datos_test, clases_test, cambio_alfa=True, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES)
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


        # ALGORITMO COA_Caotico_Cambio-Alfa
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
        print('ALGORITMO COA_Caotico_Cambio-Alfa')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = COA(datos_train, clases_train, datos_test, clases_test, caos=True, cambio_alfa=True, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES)
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


        # ALGORITMO COA_Caotico-Memetico (Best)
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
        print('ALGORITMO COA_Caotico_Memetico (Best)')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = COA_memetico(datos_train, clases_train, datos_test, clases_test, AM_Best, caos=True, npacks=N_PACKS, coyotespp=COYOTES_POR_PACK, max_evaluaciones=N_MAX_EVALUACIONES)
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