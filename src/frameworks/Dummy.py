class Dummy:
	def __init__(self, problem, EMO):
		self.problem = problem
		self.EMO = EMO

	def run(self):
		self.EMO.problem = self.problem
		self.EMO.evaluations = 0
		self.EMO.execute()
		return self.EMO.population