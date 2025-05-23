import utileria as ut
import arboles_numericos as an
import os
import random

# Descarga y descomprime los datos

url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
archivo = "datos/cancer.zip"
archivo_datos = "datos/wdbc.data"
atributos = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

# Descarga datos
if not os.path.exists("datos"):
    os.makedirs("datos")
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo)

#Extrae datos y convierte a numericos
datos = ut.lee_csv(
    archivo_datos,
    atributos=atributos,
    separador=","
)
for d in datos:
    d['Diagnosis'] = 1 if d['Diagnosis'] == 'M' else 0
    for i in range(1, 31):
        d[f'feature_{i}'] = float(d[f'feature_{i}'])
    del(d['ID'])

# Selecciona los artributos
target = 'Diagnosis'
atributos = [target] + [f'feature_{i}' for i in range(1, 31)]

# Selecciona un conjunto de entrenamiento y de validación
random.seed(42)
random.shuffle(datos)
N = int(0.8*len(datos))
print(N)
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

# Para diferentes profundidades
errores = []
for profundidad in [1, 3, 5, 10, 15, 20, 30]:
    arbol = an.entrena_arbol(
        datos_entrenamiento, 
        target, 
        atributos, 
        max_profundidad=profundidad
    )
    error_en_muestra = an.evalua_arbol(arbol, datos_entrenamiento, target)
    error_en_validacion = an.evalua_arbol(arbol, datos_validacion, target)
    errores.append( (profundidad, error_en_muestra, error_en_validacion) )
    
# Muestra los errores
print('d'.center(10) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 40)
for profundidad, error_entrenamiento, error_validacion in errores:
    print(
        f'{profundidad}'.center(10) 
        + f'{error_entrenamiento:.2f}'.center(15) 
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 40 + '\n')

#Entrena con la mejor profundidad
arbol = an.entrena_arbol(datos, target, atributos, max_profundidad=3)
error = an.evalua_arbol(arbol, datos_entrenamiento, target)
print(f'Error del modelo seleccionado entrenado con TODOS los datos: {error:.2f}')
an.imprime_arbol(arbol)

    