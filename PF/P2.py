import sys
import numpy as np
import timeit
import copy
from tabulate import tabulate

import faux

# PARÁMETROS
ALPHA = 0.8
ALPHA_BLX = 0.3
DELTA = 0.3
TAM_POBLACION = 50
N_MAX_EVALUACIONES = 15000
PCRUCE_AGG = 0.7
PMUTACION = 0.1
FRECUENCIA_MEMETICO = 10
PLS = 0.1

# Modificado P2
def mov(pesos, indice, delta):
    pesos[indice] = pesos[indice] + np.random.normal(loc = 0, scale = delta)
    if pesos[indice] > 1: pesos[indice] = 1
    if pesos[indice] < 0: pesos[indice] = 0
    return pesos

# CLASES Y FUNCIONES NECESARIAS COMUNES
class cromosoma:
    def __init__(self, datos, clases, w = []):
        if len(w) == 0:
            self.w = np.random.uniform(0,1, len(datos[0]))
        else:
            self.w = w.copy()

        self.tclas = faux.tasa_clas(self.w, datos, clases, datos, clases)
        self.tred = faux.tasa_red(self.w)
        self.tfit = ALPHA * self.tclas + (1-ALPHA) * self.tred

# OPERADOR CRUCE BLX
def cruce_BLX(c1, c2, datos, clases):
    cmax = max(c1.w + c2.w)
    cmin = min(c1.w + c2.w)
    a = cmin - (cmax-cmin)*ALPHA_BLX
    b = cmax + (cmax-cmin)*ALPHA_BLX

    h1 = np.random.uniform(a,b,len(c1.w))
    h2 = np.random.uniform(a,b,len(c2.w))

    h1[h1 < 0] = 0
    h1[h1 > 1] = 1
    h2[h2 < 0] = 0
    h2[h2 > 1] = 1

    return cromosoma(datos,clases,h1), cromosoma(datos,clases,h2)

# OPERADOR CRUCE ARITMÉTICO
def cruce_aritmetico(c1, c2, datos, clases):
    a = np.random.uniform(0,1)
    b = np.random.uniform(0,1)

    h1 = a * np.array(c1.w) + (1-a) * np.array(c2.w)
    h2 = b * np.array(c1.w) + (1-b) * np.array(c2.w)

    h1[h1 < 0] = 0
    h1[h1 > 1] = 1
    h2[h2 < 0] = 0
    h2[h2 > 1] = 1

    return cromosoma(datos,clases,h1), cromosoma(datos,clases,h2)

# OPERADOR DE SELECCIÓN
def torneo_binario(poblacion):
    id1 = np.random.randint(0, len(poblacion)-1)
    id2 = np.random.randint(0, len(poblacion)-1)

    if poblacion[id1].tfit > poblacion[id2].tfit:
        return poblacion[id1]
    else:
        return poblacion[id2]

# ALGORITMO GENÉTICO GENERACIONAL
def AGG(datos_train, clases_train, datos_test, clases_test, operador_cruce):
    generacion = 1
    poblacion = []

    # generamos la poblacion
    for i in range(TAM_POBLACION):
        poblacion.append(cromosoma(datos_train,clases_train))

    it = TAM_POBLACION

    # cromosomas a cruzar
    n_cruzar = int(PCRUCE_AGG * (len(poblacion)/2))
    # genes a mutar
    n_mutar = max(int(PMUTACION * len(poblacion)), 1)

    while it < N_MAX_EVALUACIONES:
        # cromosoma con mejor tfit
        tfits = [elem.tfit for elem in poblacion]
        mejor_id = np.argmax(tfits)

        # operador de seleccion
        nueva_poblacion = [torneo_binario(poblacion) for i in range(len(poblacion))]

        # operador de cruce
        for i in range(n_cruzar*2):
            id1 = np.random.randint(0, len(nueva_poblacion)-1)
            id2 = np.random.randint(0, len(nueva_poblacion)-1)
            h1,h2 = operador_cruce(nueva_poblacion[id1], nueva_poblacion[id2], datos_train, clases_train)
            nueva_poblacion[id1] = h1
            nueva_poblacion[id2] = h2

            i = i + 1
            it = it + 2

        poblacion_mutada = nueva_poblacion.copy()

        # para evitar repeticion
        posibilidades = [i for i in range(len(poblacion))]

        # operador de mutacion
        for i in range(n_mutar):
            id = np.random.randint(0, len(posibilidades)-1)
            id_c = posibilidades[id]
            id_g = np.random.randint(0, len(poblacion[0].w)-1)
            nuevo_c = cromosoma(datos_train, clases_train, mov(poblacion_mutada[id_c].w, id_g, DELTA))
            poblacion_mutada[id_c] = nuevo_c
            posibilidades.pop(id)
            it = it + 1

        nueva_poblacion = poblacion_mutada.copy()

        # mejor cromosoma actual
        tfits = [elem.tfit for elem in nueva_poblacion]
        mejor_id_actual = np.argmax(tfits)

        # elitismo
        if nueva_poblacion[mejor_id_actual].tfit < poblacion[mejor_id].tfit:
            tfits = [elem.tfit for elem in nueva_poblacion]
            id_peor = np.argmin(tfits)
            nueva_poblacion = np.delete(nueva_poblacion, id_peor)
            nueva_poblacion = np.append(nueva_poblacion, poblacion[mejor_id])

        poblacion = nueva_poblacion.copy()

        # mejor cromosoma actual
        tfits = [elem.tfit for elem in nueva_poblacion]
        mejor_id_actual = np.argmax(tfits)

        generacion = generacion + 1
        #if generacion % 20 == 0:
        #    print('Generacion:', generacion)

    # resultados
    w_final = poblacion[mejor_id_actual].w
    tclas = faux.tasa_clas(w_final, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(w_final)
    tfit = ALPHA*tclas + (1-ALPHA)*tred

    return tclas, tred, tfit

# ALGORITMO GENÉTICO ESTACIONARIO
def AGE(datos_train, clases_train, datos_test, clases_test, operador_cruce):
    generacion = 1
    poblacion = []

    # generamos la poblacion
    for i in range(TAM_POBLACION):
        poblacion.append(cromosoma(datos_train,clases_train))

    it = TAM_POBLACION

    # genes a mutar
    n_mutar = max(int(PMUTACION * len(poblacion)), 1)

    while it < N_MAX_EVALUACIONES:
        # padres
        nuevos_padres = [torneo_binario(poblacion), torneo_binario(poblacion)]

        # cruce
        nuevos_padres[0], nuevos_padres[1] = operador_cruce(nuevos_padres[0], nuevos_padres[1], datos_train, clases_train)
        it = it + 2

        # mutacion con probabilidad p
        for i in range(2):
            p = np.random.rand()
            if p < PMUTACION:
                id_g = np.random.randint(0, len(poblacion[0].w)-1)
                nuevo_c = cromosoma(datos_train, clases_train, mov(nuevos_padres[i].w, id_g, delta=DELTA))
                nuevos_padres[i] = nuevo_c
                it = it + 1

        # reemplazamiento
        poblacion = np.append(poblacion, nuevos_padres[0])
        poblacion = np.append(poblacion, nuevos_padres[1])
        # al ordenarlos por valor de tfit, eliminamos los dos ultimos
        poblacion = sorted(poblacion, key = lambda x : x.tfit)
        poblacion = np.delete(poblacion, 0)
        poblacion = np.delete(poblacion, 0)
        np.random.shuffle(poblacion)

        tfits = [elem.tfit for elem in poblacion]
        mejor_id_actual = np.argmax(tfits)

        generacion = generacion + 1
        #if generacion % 20 == 0:
        #    print('Generacion:', generacion)

    # resultados
    w_final = poblacion[mejor_id_actual].w
    tclas = faux.tasa_clas(w_final, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(w_final)
    tfit = ALPHA*tclas + (1-ALPHA)*tred

    return tclas, tred, tfit
    
# BUSQUEDA LOCAL BAJA INTENSIDAD
def busqueda_local_bi(datos_train, clases_train, cromosoma):
    mejor_tfit = cromosoma.tfit
    n = len(cromosoma.w)
    indices = np.arange(len(cromosoma.w))
    
    it = 0
    while it < 2*n:
        np.random.shuffle(indices)
        for indice in indices:
            it = it + 1
            if it >= 2*n: break
            copia = copy.deepcopy(cromosoma)
            copia.w = mov(copia.w, indice, delta=DELTA)
            copia.tred = faux.tasa_red(copia.w)
            copia.tclas = faux.tasa_clas(copia.w, datos_train, clases_train, datos_train, clases_train)
            copia.tfit = ALPHA * copia.tclas + (1-ALPHA) * copia.tred
            if (copia.tfit > mejor_tfit):
                cromosoma = copia
                mejor_tfit = copia.tfit
                break

    return it, cromosoma

# ALGORITMO MEMETICO GENERICO
def AM(datos_train, clases_train, datos_test, clases_test, operador_cruce, tipo_memetico):
    generacion = 1
    poblacion = []

    # generamos la poblacion
    for i in range(TAM_POBLACION):
        poblacion.append(cromosoma(datos_train,clases_train))

    it = TAM_POBLACION

    # cromosomas a cruzar
    n_cruzar = int(PCRUCE_AGG * (len(poblacion)/2))
    # genes a mutar
    n_mutar = max(int(PMUTACION * len(poblacion)), 1)

    while it < N_MAX_EVALUACIONES:
        # cromosoma con mejor tfit
        tfits = [elem.tfit for elem in poblacion]
        mejor_id = np.argmax(tfits)

        # operador de seleccion
        nueva_poblacion = [torneo_binario(poblacion) for i in range(len(poblacion))]

        # operador de cruce
        for i in range(n_cruzar*2):
            id1 = np.random.randint(0, len(nueva_poblacion)-1)
            id2 = np.random.randint(0, len(nueva_poblacion)-1)
            h1,h2 = operador_cruce(nueva_poblacion[id1], nueva_poblacion[id2], datos_train, clases_train)
            nueva_poblacion[id1] = h1
            nueva_poblacion[id2] = h2

            i = i + 1
            it = it + 2

        poblacion_mutada = nueva_poblacion.copy()

        # para evitar repeticion
        posibilidades = [i for i in range(len(poblacion))]

        # operador de mutacion
        for i in range(n_mutar):
            id = np.random.randint(0, len(posibilidades)-1)
            id_c = posibilidades[id]
            id_g = np.random.randint(0, len(poblacion[0].w)-1)
            nuevo_c = cromosoma(datos_train, clases_train, mov(poblacion_mutada[id_c].w, id_g, DELTA))
            poblacion_mutada[id_c] = nuevo_c
            posibilidades.pop(id)
            it = it + 1

        nueva_poblacion = poblacion_mutada.copy()

        # mejor cromosoma actual
        tfits = [elem.tfit for elem in nueva_poblacion]
        mejor_id_actual = np.argmax(tfits)

        # elitismo
        if nueva_poblacion[mejor_id_actual].tfit < poblacion[mejor_id].tfit:
            tfits = [elem.tfit for elem in nueva_poblacion]
            id_peor = np.argmin(tfits)
            nueva_poblacion = np.delete(nueva_poblacion, id_peor)
            nueva_poblacion = np.append(nueva_poblacion, poblacion[mejor_id])

        poblacion = nueva_poblacion.copy()

        # busqueda local de baja intensidad
        if (generacion % FRECUENCIA_MEMETICO) == 0:
            j,poblacion = tipo_memetico(poblacion, datos_train, clases_train)
            it = it + j

        # mejor cromosoma actual
        tfits = [elem.tfit for elem in nueva_poblacion]
        mejor_id_actual = np.argmax(tfits)

        generacion = generacion + 1
        #if generacion % 20 == 0:
        #    print('Generacion:', generacion)

    # resultados
    w_final = poblacion[mejor_id_actual].w
    tclas = faux.tasa_clas(w_final, datos_test, clases_test, datos_train, clases_train)
    tred = faux.tasa_red(w_final)
    tfit = ALPHA*tclas + (1-ALPHA)*tred

    return tclas, tred, tfit

# ALGORITMO MEMETICO A TODOS LOS INDIVIDUOS
def AM_All(poblacion, datos_train, clases_train):
    it = 0
    nueva_poblacion = []
    for elem in poblacion:
        j,nuevo_c = busqueda_local_bi(datos_train, clases_train, elem)
        it = it + j
        nueva_poblacion.append(nuevo_c)

    return it,nueva_poblacion

# ALGORITMO MEMETICO A SOLUCION ALEATORIA CON P=0.1
def AM_Rand(poblacion, datos_train, clases_train):
    it = 0
    indices = np.random.choice(len(poblacion),int(PLS*len(poblacion)),replace=False)
    for indice in indices:
        j,poblacion[indice] = busqueda_local_bi(datos_train,clases_train,poblacion[indice])
        it = it + j
    
    return it, poblacion

# ALGORITMO MEMETICO A 0.1*N MEJORES
def AM_Best(poblacion, datos_train, clases_train):
    it = 0
    poblacion = sorted(poblacion, key = lambda x : x.tfit, reverse=True)
    for i in range(int(0.1*len(poblacion))):
        j,poblacion[i] = busqueda_local_bi(datos_train, clases_train, poblacion[i])
        it = it + j

    np.random.shuffle(poblacion)
    
    return it, poblacion

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
        alpha = ALPHA

        # INICIALIZAMOS SEMILLA
        np.random.seed(int(sys.argv[6]))

        # PARA CONSTRUIR LA TABLA
        tabla = [['','tasa_clas','tasa_red','tasa_fitness','tiempo']]
        datos_todos = [datos1,datos2,datos3,datos4,datos5]
        clases_todos = [clases1,clases2,clases3,clases4,clases5]

        
        # ALGORITMO AGG-BLX
        tclas_medio = 0
        tred_medio = 0
        tfit_medio = 0
        tiempo_medio = 0
        tclas_std = 0
        tred_std = 0
        tfit_std = 0
        tiempo_std = 0
        print('')
        print('ALGORITMO AGG-BLX')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = AGG(datos_train, clases_train, datos_test, clases_test, cruce_BLX)
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


        # ALGORITMO AGG-Arit
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
        print('ALGORITMO AGG-Arit')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = AGG(datos_train, clases_train, datos_test, clases_test, cruce_aritmetico)
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
        

        # ALGORITMO AGE-BLX
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
        print('ALGORITMO AGE-BLX')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = AGE(datos_train, clases_train, datos_test, clases_test, cruce_BLX)
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


        # ALGORITMO AGE-Arit
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
        print('ALGORITMO AGE-Arit')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = AGE(datos_train, clases_train, datos_test, clases_test, cruce_aritmetico)
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


        # ALGORITMO AM-All
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
        print('ALGORITMO AM-All')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = AM(datos_train, clases_train, datos_test, clases_test, cruce_BLX, AM_All)
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

        # ALGORITMO AM-Rand
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
        print('ALGORITMO AM-Rand')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = AM(datos_train, clases_train, datos_test, clases_test, cruce_BLX, AM_Rand)
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

        # ALGORITMO AM-Best
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
        print('ALGORITMO AM-Best')
        for i in range(len(datos_todos)):
            # elegimos los datos de la particion
            datos_train = np.concatenate(datos_todos[:i] + datos_todos[i+1:])
            clases_train = np.concatenate(clases_todos[:i] + clases_todos[i+1:])
            datos_test = datos_todos[i]
            clases_test = clases_todos[i]
            # clasificacion + tiempo
            inicio = timeit.default_timer()
            tclas, tred, tfit = AM(datos_train, clases_train, datos_test, clases_test, cruce_BLX, AM_Best)
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