import math

class QualityIndicator(object):
  def __init__(self, referenceFront):
    self.referenceFront = referenceFront


  def euclideanDistance(self, a, b):
    dist = 0
    for i in range(len(a)):
      dist += (a[i] - b[i])*(a[i] - b[i])
    return math.sqrt(dist)

  def distanceToClosestPoint(self, point, front, distance):
    minDistance = math.inf
    for frontPoint in front:
      dist        = distance(point, frontPoint)
      minDistance = min(minDistance, dist)
      
    return minDistance

  def calculate(self, front):
    pass

class IGD(QualityIndicator):
  def __init__(self, referenceFront, mult_pow=2.0):
    super(type(self), self).__init__(referenceFront)
    self.mult_pow = mult_pow

  def calculate(self, front):
    if len(front) == 0 or len(self.referenceFront) == 0:
      print("IDG evaluate: front or referenceFront without elements")

    sum_ = 0
    for point in self.referenceFront:
      dist = self.distanceToClosestPoint(point=point,
                                         front=front,
                                         distance=self.euclideanDistance)
      sum_ += math.pow(dist, self.mult_pow)

    sum_ = math.pow(sum_, 1.0/self.mult_pow)
    
    return sum_ / len(self.referenceFront)
  
class GD(QualityIndicator):
  def __init__(self, referenceFront, mult_pow=2.0):
    super(type(self), self).__init__(referenceFront)
    self.mult_pow = mult_pow

  def calculate(self, front):
    if len(front) == 0 or len(self.referenceFront) == 0:
      print("IDG evaluate: front or referenceFront without elements")

    sum_ = 0
    for point in front:
      dist = self.distanceToClosestPoint(point=point,
                                         front=self.referenceFront,
                                         distance=self.euclideanDistance)
      sum_ += math.pow(dist, self.mult_pow)
      
    sum_ = math.pow(sum_,1.0/self.mult_pow)
    
    return sum_ / len(front)