import utileria as ut
import arboles_cualitativos as ac
import os
import random

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

print(datos)