# Problemas, Operadores e Métricas

Para assegurar a confiabilidade, versatilidade e testar o desempenho dos algoritmos desenvolvidos, o **PyManyObjective** possui a implementação nativa de problemas padronizados da literatura, operadores evolutivos focados em espaços reais contínuos e métodos de mensuração consagradas da computação.

Abaixo temos os detalhes focados nas implementações isoladas que compõem os argumentos injetados nos frameworks de busca.

## Operadores Implementados

Os operadores são os motores da busca nos algoritmos evolucionários. Eles são responsáveis por criar variação nos genes, realizar pressões de convergência ou de seleção, para conduzir a população gradualmente para áreas promissoras do espaço de busca global.

### Cruzamento (Crossover)
O framework foca na representação contínua das variáveis (espaço Real).
- **SBX Crossover (Simulated Binary Crossover)**: Combina os valores quantitativos de cada variável de duas soluções (pais). O quão perto ou longe os filhos gerados estarão geneticamente dos pais é controlado e ditado pelo "índice de distribuição" (`distributionIndex`). Possui maior probabilidade de gerar filhos focados em explotação.

### Mutação (Mutation)
- **Polynomial Mutation**: O algoritmo de mutação polinomial adiciona pequenos distúrbios aos valores atuais das variáveis da solução. Semelhante ao SBX, se baseia num modelo de decaimento polinomial que utiliza um índice de distribuição específico para guiar os pulos da variável limitando as perturbações agressivas em demasia.

### Seleção e Esparsidade (Selection & Sparsity)
- **Binary Tournament (Seleção)**: Seleciona aleatoriamente duas soluções dentro da população e realiza um torneio par-a-par. A solução vencedora é invariavelmente a que domina a outra pelo conceito de Pareto (ou com base no seu critério de esparsidade no cenário em que não haja dominância aparente).
- **Crowding Distance (Sparsity)**: Mede a densidade circundante das soluções no espaço para priorizar nas frentes aquelas que estão mais isoladas a fim de manter a variabilidade genética da pesquisa.

---

## Benchmarks de Otimização (Problemas)

Com o propósito de provar a eficiência nos cenários complexos do mundo real, funções estressantes focadas na geometria da busca foram implementadas. 

### Família ZDT
Proposta por Zitzler, Deb e Thiele (2000), essas funções possuem o *baseline* de dois objetivos e testam os diferentes desafios presentes no ambiente:
- **ZDT1**: Possui fronteira de Pareto convexa e formato simples. Usado comumente para aferir a convergência bruta básica.
- **ZDT3**: Sua fronteira é desconexa (discreta), sendo composta por múltiplos segmentos quebrados que atrapalham a frente final.
- **ZDT6**: Problema não uniforme e robusto, possuindo uma densidade variável não constante e fronteira não convexa.

### Família DTLZ
Proposta por Deb, Thiele, Laumanns e Zitzler (2002), essa famosa suíte foi desenhada e expandida especialmente para validação das qualidades dos algoritmos perante Problemas com Muitos Objetivos (MaOPs - *Many-Objective*), os quais podem comportar de 3 a N dimensões numéricas:
- **DTLZ1**: Apresenta superfície linear hiperplana. Contém uma extraordinária dificuldade de convergência ao alvo devido à brutal quantidade de ótimos locais produzidos por sua função combinatória restritiva (g(x)).
- **DTLZ2**: Traz a exigência de uma fronteira esférica (côncava). Ideal para atestar a capacidade final que um algoritmo tem de cobrir suavemente toda a superfície da área de Pareto sem agrupamentos esparsos.
- **DTLZ3**: Altamente multimodal e traiçoeiro. Combina efetivamente o corpo côncavo do DTLZ2 com os vales de ótimos locais perigosos do DTLZ1. Excelente para provar o escape local de um framework.
- **DTLZ4**: Modifica o DTLZ2 para dificultar a manutenção de um conjunto simétrico espalhado, apresentando uma distorção maciça de densidade de mapeamento próxima às beiras e quinas da superfície de Pareto.
- **DTLZ5 e DTLZ6**: Instâncias únicas de fronteira de Pareto degenerada (unidimensional ou de sub-dimensão estrita em relação ao grande número de objetivos solicitados).
- **DTLZ7**: Possui a fronteira de Pareto de características totalmente desconectadas (composta por múltiplas ilhas ou regiões distintas isoladas, exigindo que os indivíduos saltem para diferentes quadrantes da busca).

---

## Métricas de Qualidade

A medição do balanço e da performance final se assenta simultaneamente em dois fortes pilares da literatura: na **convergência** (quão aproximadas do alvo verídico real as soluções encontradas estão?) e na **diversidade/extensão** (quão bem e homogeneamente espalhadas pelos cantos as soluções estão distribuídas pela fronteira de Pareto?).

- **IGD (Inverted Generational Distance):** É a medida mais confiável do momento. Quantifica o erro extraindo a distância média euclidiana da verdadeira e infinita Fronteira de Pareto em direção à fronteira alcançada de soluções obtidas pelo algoritmo. Exige conhecimento *a priori* do arquivo ideal do problema. *Quanto menor e mais próximo de 0 for o valor, melhor a convergência e distribuição combinadas do seu achado.*
- **GD (Generational Distance):** Inverte completamente o sentido do cálculo de distâncias do IGD, sendo feito partindo da fronteira que seu algoritmo alcançou para os pontos da fronteira ideal em si. Avalia puramente a convergência da força bruta de seu modelo, não recompensando a abrangência uniforme do formato.
- **Hypervolume (HV):** Calcula efetivamente o hipervolume global (volume multi-dimensional no espaço) da região espacial que as suas soluções dominam por baixo da curva, usando uma âncora imaginária denominada ponto de referência *Nadir* (o pior limite viável). *Neste indicador em particular, valores absolutos maiores indicam uma maximização perfeita de performance na convergência e diversidade de soluções obtidas.*