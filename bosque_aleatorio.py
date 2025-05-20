from collections import Counter
from arboles_numericos import entrena_arbol, NodoN
import random

def entrena_bosque_aleatorio(datos, target, clase_default, 
                       num_arboles=10, 
                       porcentaje_variables=0.5,
                       max_profundidad=None, 
                       acc_nodo=1.0, 
                       min_ejemplos=0):
    """
    Entrena un bosque aleatorio usando el algoritmo de arboles 
    
    Parámetros:
    -----------
    datos: list(dict)
        Una lista de diccionarios donde cada diccionario representa una instancia
    target: str
        El nombre del atributo que se quiere predecir
    clase_default: str
        El valor de la clase por default
    num_arboles: int
        Numero de árboles a entrenar
    porcentaje_variables: float
        Porcentaje de variables a considerar en cada nodo (entre 0 y 1)
    max_profundidad: int
        La máxima profundidad de cada árbol. Si es None, no hay límite
    acc_nodo: float
        El porcentaje de acierto mínimo para considerar un nodo como hoja
    min_ejemplos: int
        El numero mínimo de ejemplos para considerar un nodo como hoja
        
    Regresa:
    --------
    bosque: list(Nodo)
        Una lista con los nodos raiz de cada árbol en el bosque
    """

    atributos = list(datos[0].keys())
    atributos.remove(target)
    num_variables = max(1, int(len(atributos) * porcentaje_variables))
    
    bosque = []
    
    for _ in range(num_arboles):
        # Creamos un subset de datos con muestreo con reemplazo 
        indices = random.choices(range(len(datos)), k=len(datos))
        subset_datos = [datos[i] for i in indices]
        
        # Entrenamos un arbol con ese subset de datos
        # y con un número limitado de variables en cada nodo
        arbol = entrena_arbol(
            subset_datos, 
            target, 
            clase_default,
            max_profundidad=max_profundidad, 
            acc_nodo=acc_nodo, 
            min_ejemplos=min_ejemplos,
            variables_seleccionadas=num_variables  # Usamos el entero para selección aleatoria
        )
        
        bosque.append(arbol)
    
    return bosque

def predice_instancia_arbol(arbol, instancia): 
    nodo = arbol
    while not nodo.terminal:
        if instancia.get(nodo.atributo, float('inf')) < nodo.valor:
            nodo = nodo.hijo_menor
        else:
            nodo = nodo.hijo_mayor
            
    return nodo.clase_default

def predice_bosque(bosque, instancia): 
    # Obtenemos las predicciones de cada arbol
    predicciones = [predice_instancia_arbol(arbol, instancia) for arbol in bosque]
    
    # Votación mayoritaria
    contador = Counter(predicciones)
    return contador.most_common(1)[0][0]

def evalua_bosque(bosque, datos_prueba, target):
    correcto = 0
    
    for instancia in datos_prueba:
        prediccion = predice_bosque(bosque, instancia)
        if prediccion == instancia[target]:
            correcto += 1
            
    return correcto / len(datos_prueba) if datos_prueba else 0