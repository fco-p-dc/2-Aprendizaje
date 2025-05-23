import utileria as ut
import bosque_aleatorio as ba
import os
import random
from collections import defaultdict

# Descarga y descomprime los datos

url = 'https://archive.ics.uci.edu/static/public/2/adult.zip'
archivo = 'datos/adults.zip'
archivo_datos = 'datos/adult.data'
atributos = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
target = 'income'

if not os.path.exists('datos'):
    os.makedirs('datos')
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo)

# Lee los datos
datos = ut.lee_csv(archivo_datos, atributos=atributos, separador=',')


# Convierte a numéricos y transforma atributos categóricos
diccionarios = defaultdict(dict)
valores_usados = defaultdict(set)

# Recolecta valores únicos por atributo categórico
atributos_categoricos = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

# Etiquetas numéricas
atributos_numericos = [
    'age', 'fnlwgt', 'education-num', 'capital-gain',
    'capital-loss', 'hours-per-week'
]

datos = [
    d for d in datos
    if all(k in d and d[k] != ' ?' and d[k] != '?' for k in atributos)
]

# Paso 1: Recolecta todos los valores únicos por atributo categórico
for d in datos:
    for atributo in atributos_categoricos:
        valores_usados[atributo].add(d[atributo])

# Paso 2: Asigna un número entero a cada categoría
for atributo in atributos_categoricos:
    for idx, valor in enumerate(sorted(valores_usados[atributo])):
        diccionarios[atributo][valor] = idx

# Paso 3: Aplica transformaciones
for d in datos:
    d['income'] = 1 if '>50' in d['income'].strip() else 0
    for atributo in atributos_categoricos:
        d[atributo] = diccionarios[atributo][d[atributo]]
    for atributo in atributos_numericos:
        d[atributo] = float(d[atributo])

#print(datos)

random.seed(42)
random.shuffle(datos)
N = int(0.8*len(datos))
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

# TOTAL DE DATOS 24,000 ENTRENA CON PRECAUCION

# Selecciona un conjunto de entrenamiento y de validación con menor tamaño
#TAMANO_MUESTRA_PRUEBAS = 0.1
#sub_datos = datos[:int(TAMANO_MUESTRA_PRUEBAS * len(datos))]
#N_sub = int(0.8 * len(sub_datos))

#datos_entrenamiento = datos[:N_sub]
#datos_validacion = datos[N_sub:]

errores = []
errores_prof = []
errores_var = []
num_arboles_opciones = [1, 3, 5, 10, 15, 20, 30]
profundidades_opciones = [1, 3, 5, 10, 15, 20, 30]
cantidad_variables_opciones = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]


# Para diferentes números de árboles
print("Numero de arboles diferentes".center(40))
for num_arboles in num_arboles_opciones:
    print(f"Entrenando bosque con {num_arboles} arboles")
    bosque = ba.entrena_bosque_aleatorio(
        datos_entrenamiento, 
        target=target,
        clase_default=0,
        num_arboles=num_arboles,
        porcentaje_variables=0.5,
        max_profundidad=10,
        acc_nodo=1.0,
        min_ejemplos=1
    )
    
    error_train = 1 - ba.evalua_bosque(bosque, datos_entrenamiento, target)
    error_val = 1 - ba.evalua_bosque(bosque, datos_validacion, target)
    
    errores.append((num_arboles, error_train, error_val))

print('d'.center(10) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 40)
for num_arboles, error_train, error_val in errores:
    print(
        f'{num_arboles}'.center(10) 
        + f'{error_train:.2f}'.center(15) 
        + f'{error_val:.2f}'.center(15)
    )
print('-' * 40 + '\n')

# Para diferentes profundidades
print("Profundidades diferentes".center(40))
for profundidad in profundidades_opciones:
    print(f"Entrenando bosque con {profundidad} profundidad")
    bosque = ba.entrena_bosque_aleatorio(
        datos_entrenamiento, 
        target=target,
        clase_default=0,
        num_arboles=10,
        porcentaje_variables=0.5,
        max_profundidad=profundidad,
        acc_nodo=1.0,
        min_ejemplos=1
    )
    
    error_train = 1 - ba.evalua_bosque(bosque, datos_entrenamiento, target)
    error_val = 1 - ba.evalua_bosque(bosque, datos_validacion, target)
    
    errores_prof.append((profundidad, error_train, error_val))

print('d'.center(10) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 40)
for profundidad, error_train, error_val in errores_prof:
    print(
        f'{profundidad}'.center(10) 
        + f'{error_train:.2f}'.center(15) 
        + f'{error_val:.2f}'.center(15)
    )
print('-' * 40 + '\n')

# Para diferentes numero de variables
print("Variables diferentes".center(40))
for variables in cantidad_variables_opciones:
    print(f"Entrenando bosque con {variables} porcentaje de variables")
    bosque = ba.entrena_bosque_aleatorio(
        datos_entrenamiento, 
        target=target,
        clase_default=0,
        num_arboles=10,
        porcentaje_variables=variables,
        max_profundidad=10,
        acc_nodo=1.0,
        min_ejemplos=1
    )
    
    error_train = 1 - ba.evalua_bosque(bosque, datos_entrenamiento, target)
    error_val = 1 - ba.evalua_bosque(bosque, datos_validacion, target)
    
    errores_var.append((variables, error_train, error_val))

print('d'.center(10) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 40)
for variables, error_train, error_val in errores_var:
    print(
        f'{variables}'.center(10) 
        + f'{error_train:.2f}'.center(15) 
        + f'{error_val:.2f}'.center(15)
    )
print('-' * 40 + '\n')

#Entrena con los mejores parámetros
mejorNumArboles = 15
mejorProfundidad = 15
mejorPorcentajeVariables = 0.5

mejorBosque = ba.entrena_bosque_aleatorio(
    datos_entrenamiento,
    target=target,
    clase_default=0,
    num_arboles=mejorNumArboles,
    porcentaje_variables=mejorPorcentajeVariables,
    max_profundidad=mejorProfundidad,
    acc_nodo=1.0,
    min_ejemplos=1
    )
error_trainMB = 1 - ba.evalua_bosque(mejorBosque, datos_entrenamiento, target)
error_valMB = 1 - ba.evalua_bosque(mejorBosque, datos_validacion, target)
print(f'Error del modelo entrenamiento seleccionado entrenado con N datos: {error_trainMB:.2f}')
print(f'Error del modelo validacion seleccionado entrenado con N los datos: {error_valMB:.2f}')