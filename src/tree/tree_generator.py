import random

from tree.config import FUNCTION_SET
from tree.node import Node
from tree.tree_util import get_depth


class BinaryTreeGenerator:
    def __init__(self, terminal_set: list):
        self._terminal_set: list = terminal_set
        self._function_set: list = FUNCTION_SET

    def full(self, depth: int) -> Node:
        if depth > 1:
            return Node(
                value=random.choice(self._function_set),
                left=self.full(depth=depth - 1),
                right=self.full(depth=depth - 1)
            )
        else:
            return Node(value=random.choice(self._terminal_set))

    def grow(self, depth: int) -> Node:
        if depth > 1:
            if random.randint(0, 1) == 0:
                return Node(
                    value=random.choice(self._function_set),
                    left=self.grow(depth=depth - 1),
                    right=self.grow(depth=depth - 1)
                )
            else:
                return Node(value=random.choice(self._terminal_set))
        else:
            return Node(value=random.choice(self._terminal_set))

    def ramped(self, depth: int, index_on_level: int) -> Node:
        if index_on_level % 2 == 0:
            return self.full(depth=depth)
        else:
            tree = self.grow(depth=depth)
            while get_depth(tree) != depth:
                tree = self.grow(depth=depth)
            return tree
