#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:22 2021

@author: jad
"""

from src.NaryTournament import NaryTournament

# Classe de torneio binário
class BinaryTournament(NaryTournament):
  def __init__(self):
    super(BinaryTournament, self).__init__(2)