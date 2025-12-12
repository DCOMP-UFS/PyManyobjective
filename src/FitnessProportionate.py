#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gustaaragao
"""

from src.Selection import Selection

# -1 * FITNESS ou 1 / FITNESS na Minimização

class FitnessProportionate(Selection):
    def __init__(self):
        super().__init__()

    def select(solutions):
        if (len(solutions) == 0):
            return None
        
