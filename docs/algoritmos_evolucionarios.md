# Algoritmos Evolucionários

O framework traz a implementação nativa de importantes algoritmos evolucionários multiobjetivo (MOEAs), amplamente conhecidos na literatura para solucionar MOPs e MaOPs.

## NSGA-II
O *Non-dominated Sorting Genetic Algorithm II* (NSGA-II) é um algoritmo evolucionário que se destaca por sua seleção de soluções baseada em ordenação por não-dominância rápida (*fast non-dominated sorting*) e no cálculo da distância de aglomeração (*crowding distance*). O primeiro cria grupos de dominância (fronteiras) e o segundo garante a diversidade das soluções da fronteira, favorecendo soluções mais isoladas.

## NSGA-III
Evolução do NSGA-II projetada para lidar com Problemas de Otimização com Muitos Objetivos (MaOPs - 4 ou mais objetivos). Ele substitui o *crowding distance* por uma abordagem baseada em pontos de referência (*reference points*). Essa abordagem permite uma melhor convergência e distribuição das soluções em espaços de alta dimensão, superando a ineficiência do NSGA-II tradicional nesses cenários.

## MOEA/D
O *Multiobjective Evolutionary Algorithm Based on Decomposition* (MOEA/D) é um algoritmo que decompõe o problema multiobjetivo em diversos subproblemas de otimização escalar (com apenas um objetivo) e os resolve simultaneamente, fazendo uso de vetores de peso.