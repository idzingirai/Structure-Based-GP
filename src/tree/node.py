class Node:
    def __init__(self, value, left=None, right=None, fitness=0.0):
        self.value: str = value
        self.left: Node = left
        self.right: Node = right
        self.fitness: float = fitness
