# Todo List

## 10/03/2026 - 23/03/2026: Fazer funcionar o DVL Framework na escolha da população inicial
- [x] Incorporar o DVL presente na branch do Everton na minha branch
- [x] Usar os experimentos propostos por Arthur
    - [x] Fazer a leitura das referências bases explicando os problemas de DTLZ
        - [x] k = n - m + 1, onde k é a dificuldade do problema
    - [x] Executar o DLTZ2 com os seguintes parâmetros:
        - [x] 3 objectives and 12 variables
        - [x] 10 objectives and 12 variables
- [x] Usar a plotagem dos pontos da região de Pareto para testar
- [x] Atualizar a documentação do projeto em /docs
- [x] Criar um utils/ com funções para plotagem

Observação:
- Os testes manuais do fluxo experimental do DVL foram centralizados em `DVLFramework.ipynb`.
- O notebook agora inclui gráficos para debug da fronteira e da população prevista pelo DVL.

```Códigos do GNUPlot
Código do GNUPLOT

=====
Fronteira do DTLZ1
====
set parametric

set urange[0:1]
set vrange[0:1]

# Parametric functions for the sphere
set xrange[0:1]
set yrange[0:1]
set zrange[0:1]

r=1
fx(v,u) = 0.5*v*u
fy(v,u) = 0.5*v*(1-u)
fz(v)   = 0.5*(1-v)

set pointsize 1

splot fx(v,u),fy(v,u),fz(v) notitle lt 9
===
Fronteira do DTLZ2
=====

set parametric

set urange[0:1.57]
set vrange[0:pi]

# Parametric functions for the sphere
set xrange[0:1]
set yrange[0:1]
set zrange[0:1]

r=1
fx(v,u) = r*cos(v)*cos(u)
fy(v,u) = r*cos(v)*sin(u)
fz(v)   = r*sin(v)

set pointsize 1

splot fx(v,u),fy(v,u),fz(v) notitle lt 9
==
Sua fronteira
===

set samples 21
set isosample 11
set xlabel "X axis" 
set ylabel "Y axis" 
set zlabel "Z axis" 
set xrange [0:1]
set yrange [0:1]
set zrange [0:1]
set xyplane(0,0) 
splot "ref_3.txt" title "Reference Points"
```
