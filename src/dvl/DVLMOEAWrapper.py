"""Wrapper de orcamento para o DVL Framework.

Executa o DVL com um orcamento fixo de avaliacoes (uma fracao X% do total, ou um
numero explicito de avaliacoes) e entao "passa o bastao" para o MOEA terminar as
avaliacoes restantes. Foi pensado para permitir definir manualmente a quantidade
de avaliacoes gastas no DVL em experimentos futuros (parametro dvl_evaluations).

Divisao do orcamento do DVL (escolha "desacoplada"):
  - dvl_budget = round(dvl_fraction * max_evaluations)  (ou dvl_evaluations)
  - n_estimated = min(dvl_budget // 2, N_pontos_referencia)  -> estimacao
  - sample      = dvl_budget - n_estimated                   -> amostra LHS

O numero de pontos estimados pelo DVL fica DESACOPLADO do numero de pontos de
referencia do MOEA: estima-se apenas quantos cabem na metade do orcamento (com
um teto no total de pontos disponiveis). Qualquer sobra do orcamento vai para a
amostra LHS (mais dados de treino -> modelo melhor). Assim o DVL consome
exatamente dvl_budget avaliacoes reais (quando dvl_budget >= 2) e o MOEA recebe
max_evaluations - dvl_budget.
"""

from src.dvl.DVL import DVLFramework, generate_reference_points


class DVLMOEAWrapper:
    def __init__(
        self,
        problem,
        moea_class,
        model,
        m,
        max_evaluations,
        dvl_fraction=None,
        dvl_evaluations=None,
        reference_point_divisions=None,
        sampling_seed=None,
        objective_transform=None,
        crossover=None,
        mutation=None,
        surrogate_class=None,
        max_training_samples=None,
        seed_moea_with_samples=False,
    ):
        self.problem = problem
        self.moea_class = moea_class
        self.model = model
        self.m = m
        self.max_evaluations = max_evaluations
        self.sampling_seed = sampling_seed
        self.objective_transform = objective_transform
        self.crossover = crossover
        self.mutation = mutation
        self.surrogate_class = surrogate_class
        self.max_training_samples = max_training_samples
        self.seed_moea_with_samples = seed_moea_with_samples

        if reference_point_divisions is None:
            reference_point_divisions = {2: 99, 3: 12, 10: 3}.get(m, 12)
        self.reference_point_divisions = reference_point_divisions

        # Orcamento do DVL (em avaliacoes reais da funcao objetivo).
        if dvl_evaluations is not None:
            dvl_budget = int(dvl_evaluations)
        elif dvl_fraction is not None:
            dvl_budget = int(round(dvl_fraction * max_evaluations))
        else:
            raise ValueError("Informe dvl_fraction ou dvl_evaluations.")
        dvl_budget = max(0, min(dvl_budget, max_evaluations))
        self.dvl_fraction = dvl_fraction
        self.dvl_budget = dvl_budget
        self.moea_budget = max_evaluations - dvl_budget

        # Numero total de pontos de referencia disponiveis para estimacao.
        n_reference_points = len(
            generate_reference_points(m, reference_point_divisions)
        )
        self.n_reference_points = n_reference_points

        # Divisao do orcamento: metade (teto = pontos disponiveis) para estimacao,
        # o restante para a amostra LHS. Sobras de orcamento engordam a amostra.
        n_estimated = min(max(1, dvl_budget // 2), n_reference_points)
        sample = dvl_budget - n_estimated
        if sample < 1 and dvl_budget >= 2:
            sample = 1
            n_estimated = dvl_budget - 1
        self.planned_sample_size = sample
        self.planned_estimated_points = n_estimated

        # Metricas preenchidas apos execute().
        self.framework = None
        self.dvl_evaluations = 0
        self.moea_evaluations = 0
        self.objective_calls = 0
        self.model_training_time = 0.0
        self.real_evaluation_time = 0.0

    def execute(self, return_trace=False):
        framework = DVLFramework(
            pop_size=self.planned_sample_size,
            max_eval=self.max_evaluations,
            ClassMoea=self.moea_class,
            model=self.model,
            problem=self.problem,
            reference_point_divisions=self.reference_point_divisions,
            sampling_seed=self.sampling_seed,
            objective_transform=self.objective_transform,
            max_estimated_points=self.planned_estimated_points,
            # O MOEA roda no tamanho natural de populacao (= nº de pontos de
            # referencia); a populacao estimada pelo DVL apenas o semeia.
            moea_population_size=self.n_reference_points,
            max_training_samples=self.max_training_samples,
            seed_moea_with_samples=self.seed_moea_with_samples,
            run_moea=True,
            surrogate_class=self.surrogate_class,
            crossover=self.crossover,
            mutation=self.mutation,
        )
        self.framework = framework

        population, trace = framework.execute(return_trace=True)

        # Avaliacoes reais de fato consumidas (problem.avaliations e a fonte da
        # verdade; o DVL gasta sample + estimados, o resto e do MOEA).
        self.dvl_evaluations = int(trace["dvl_evaluations"])
        self.objective_calls = int(trace["objective_calls"])
        self.moea_evaluations = max(0, self.objective_calls - self.dvl_evaluations)
        self.model_training_time = float(framework.model_training_time)
        self.real_evaluation_time = float(self.problem.evaluation_time)

        if return_trace:
            return population, trace
        return population
