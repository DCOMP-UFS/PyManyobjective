import math
import numpy as np
from pymoo.factory import get_performance_indicator

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

    return math.pow(sum_ / len(self.referenceFront), 1.0/self.mult_pow)
  
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

    return math.pow(sum_ / len(front), 1.0/self.mult_pow)


class HV(QualityIndicator):
  def __init__(self, referencePoint, idealPoint=None):
    super(type(self), self).__init__(referencePoint)
    self.referencePoint = np.asarray(referencePoint, dtype=float)
    if idealPoint is None:
      idealPoint = np.zeros(len(self.referencePoint), dtype=float)
    self.idealPoint = np.asarray(idealPoint, dtype=float)
    self.last_valid_count = 0
    self.last_total_count = 0
    self.indicator = get_performance_indicator("hv", ref_point=self.referencePoint)

  def _filter_front(self, front):
    front = np.asarray(front, dtype=float)

    if front.size == 0:
      self.last_valid_count = 0
      self.last_total_count = 0
      return np.empty((0, len(self.referencePoint)), dtype=float)

    if front.ndim == 1:
      front = front.reshape(1, -1)

    valid_mask = np.all(np.isfinite(front), axis=1)
    valid_mask &= np.all(front >= self.idealPoint, axis=1)
    valid_mask &= np.all(front <= self.referencePoint, axis=1)

    self.last_total_count = len(front)
    self.last_valid_count = int(np.count_nonzero(valid_mask))
    return front[valid_mask]

  def calculate(self, front) -> float:
    valid_front = self._filter_front(front)
    if len(valid_front) == 0:
      return 0.0
    return float(self.indicator.calc(valid_front))
