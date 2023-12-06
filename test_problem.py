#from tests.test_zdt import run
from tests.test_dtlz import run

P = run(10,100)
func = array([i.objectives for i in P])
Scatter(angle=(45,45)).add(func).show()