import random

from genetic_program.config import TOURNAMENT_SIZE
from tree.node import Node
from tree.tree_util import copy_tree


def tournament_selection(population: [Node]) -> Node:
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda t: t.fitness)
    return copy_tree(tournament[0])
