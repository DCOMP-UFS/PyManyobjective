# Decision Variable Learning (DVL)

Ao contrário dos modelos *Surrogate* tradicionais (que mapeiam o espaço de variáveis de decisão para o espaço de objetivos), a Modelagem Inversa mapeia os valores das funções objetivo de volta para as variáveis de decisão. 

## O Algoritmo DVL
O *Decision Variable Learning* (DVL) utiliza modelos de aprendizado de máquina para gerar diretamente um conjunto aproximado do Ótimo de Pareto (Pareto-optimal set). 
Ele é baseado em três etapas principais:
1. **Criação da amostragem inicial:** Uma amostragem é criada via *Latin Hypercube Sampling* (LHS), cobrindo o espaço de busca.
2. **Treinamento do modelo inverso:** Utilizando modelos de *machine learning* (como Regressão Linear, MLP, Random Forest ou Support Vector Regression - SVR), ele aprende o mapeamento entre os valores de função objetivo obtidos e as variáveis de decisão associadas.
3. **Predição de novas soluções:** Ele projeta pontos de referência (baseados na distribuição hiperplana normalizada de Das e Dennis) e os fornece ao modelo para estimar as variáveis de decisão de soluções que estarão idealmente muito próximas da Fronteira de Pareto.

O DVL executa este processo de forma iterativa, treinando novamente os modelos com as soluções que estão mais próximas aos pontos de referência de interesse, até atingir a convergência ou o limite de avaliações reais.

## DVL Framework
Além da sua execução isolada (*standalone*), o PyManyObjective fornece o **DVL Framework**, uma arquitetura de integração entre o DVL e qualquer MOEA clássico (como NSGA-III, MOEA/D ou RVEA).
- O DVL atua na inicialização da população: com poucas avaliações, ele aproxima a fronteira de Pareto com ótima distribuição global.
- Essa população inicial é, em seguida, passada para um MOEA que utiliza seus operadores de busca local e global para melhorar ainda mais o conjunto de soluções antes do tempo limite de processamento terminar.

Nos experimentos conduzidos em problemas da literatura (DTLZ1-7) e do mundo real (Sincronização de Semáforos utilizando o simulador SUMO), o DVL Framework demonstrou superar os algoritmos clássicos ao atingir valores de convergência maiores gastando muito menos tempo computacional.