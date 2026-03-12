# Algoritmos Baseados em Surrogates

Os algoritmos de *Surrogate-Assisted Evolutionary Algorithms* (SAEAs) utilizam modelos de aproximação computacional (*surrogates* ou metamodelos) no lugar das funções objetivo originais, visando reduzir substancialmente o tempo computacional em problemas de otimização caros e complexos.

No framework, estão presentes os seguintes algoritmos SAEAs:

## M1, M3 e M6
Modelos propostos por Kalyanmoy Deb et al. (2019) que utilizam o modelo Kriging (Processos Gaussianos) como *surrogate*.
- Eles iniciam gerando uma população aleatória.
- A cada iteração (passo), eles executam um algoritmo evolucionário (ex: NSGA-III) em que o cálculo dos objetivos é feito utilizando o modelo *surrogate*.
- O treinamento do *surrogate* ocorre a cada τ passos usando avaliações reais da função objetivo.
- **M3 e M6** convertem a priori o problema multiobjetivo em um único objetivo (via função ASF no M3, ou KKTPM no M6) antes de otimizá-lo, fazendo o uso de direções de referência.

## SMOyO e SMByO
Modelos cujas características diferem por utilizarem o *Random Forest* (Florestas Aleatórias) como modelo de *surrogate*.
- Apenas começam a utilizar os metamodelos após um número inicial predefinido de avaliações reais (`tmax`).
- A população de treino baseia-se nas fronteiras de Pareto geradas nas interações anteriores.
- **SMOyO**: Treinamento contínuo/parcial, o *surrogate* é treinado a cada iteração com a população atual.
- **SMByO**: O *surrogate* só é treinado quando a contagem atinge o `tmax`.