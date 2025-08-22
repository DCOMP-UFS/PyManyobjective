#venv/bin/python3
"""
need sorted containerspip install sortedcontainers
"""
from tests.test_dvl import run
import io
import sys

lista = []

def resultado_arq(caminho, lista_resultado):
    
    with open(caminho, "w", encoding="utf-8") as arq:
        for _ in range(1):
            arq.write(str(lista_resultado))

    lista_resultado.clear()



def resultados(lista_resultados, n_obj, evals_allowed, pop_size): 
    
    for i in range(1):
        print(f"\n{i+1}° Execução:\n")
        resultado = run(n_obj, evals_allowed, pop_size)
        lista_resultados.append(resultado)
    

# 1000, 10000, 1000000 Para 3 Objetivos
resultados(lista, 3, 1000, 100)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=1000_popsize_100", lista)

resultados(lista, 3, 1000, 150)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=1000_popsize_150", lista)

resultados(lista, 3, 1000, 200)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=1000_popsize_200", lista)
# *****************************************************************************



resultados(lista, 3, 1500, 100)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=1500_popsize_100", lista)

resultados(lista, 3, 1500, 150)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=1500_popsize_150", lista)

resultados(lista, 3, 1500, 200)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=1500_popsize_200", lista)
# *****************************************************************************



resultados(lista, 3, 10000, 100)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=10000_popsize_100", lista)

resultados(lista, 3, 10000, 150)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=10000_popsize_150", lista)

resultados(lista, 3, 10000, 200)
resultado_arq("tests/resultados_dvl/dvl_obj=3_evaluate=10000_popsize_200", lista)
# *****************************************************************************





# # 1000, 10000, 1000000 Para 10 Objetivos
# resultados(lista, 10, 1000, 100)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=1000_popsize_100", lista)

# resultados(lista, 10, 1000, 150)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=1000_popsize_150", lista)

# resultados(lista, 10, 1000, 200)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=1000_popsize_200", lista)
# # *****************************************************************************


# resultados(lista, 10, 1500, 100)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=1500_popsize_100", lista)

# resultados(lista, 10, 1500, 150)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=1500_popsize_150", lista)

# resultados(lista, 10, 1500, 200)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=1500_popsize_200", lista)
# # *****************************************************************************


# resultados(lista, 10, 10000, 100)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=10000_popsize_100", lista)

# resultados(lista, 10, 10000, 150)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=10000_popsize_150", lista)

# resultados(lista, 10, 10000, 200)
# resultado_arq("tests/resultados_dvl/dvl_obj=10_evaluate=10000_popsize_200", lista)
# # *****************************************************************************







# 10 execuções:
    # pop_size=100, max_eval=1000, obj=3
    # pop_size=150, max_eval=1000, obj=3
    # pop_size=200, max_eval=1000, obj=3

    # pop_size=100, max_eval=10000, obj=3
    # pop_size=150, max_eval=10000, obj=3
    # pop_size=200, max_eval=10000, obj=3
    
    # pop_size=100, max_eval=100000, obj=3
    # pop_size=150, max_eval=100000, obj=3
    # pop_size=200, max_eval=100000, obj=3


    # pop_size=100, max_eval=1000, obj=10
    # pop_size=150, max_eval=1000, obj=10
    # pop_size=200, max_eval=1000, obj=10

    # pop_size=100, max_eval=10000, obj=10
    # pop_size=150, max_eval=10000, obj=10
    # pop_size=200, max_eval=10000, obj=10
    
    # pop_size=100, max_eval=100000, obj=10
    # pop_size=150, max_eval=100000, obj=10
    # pop_size=200, max_eval=100000, obj=10

