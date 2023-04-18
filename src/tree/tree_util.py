import random
from typing import Optional
from collections import deque

from src.tree.node import Node


def prune(root: Node, max_depth: int, terminal_set: list) -> Optional[Node]:
    """
    Prune a tree to a given depth.
    :param root:
    :param max_depth:
    :param terminal_set:
    :return: pruned tree
    """
    if root is None:
        return None

    if max_depth == 0:
        return Node(random.choice(terminal_set))

    root.left = prune(root=root.left, max_depth=max_depth - 1, terminal_set=terminal_set)
    root.right = prune(root=root.right, max_depth=max_depth - 1, terminal_set=terminal_set)
    return root


def get_depth(root: Node) -> int:
    """
    Get the depth of a tree.
    :param root:
    :return: depth of the tree
    """
    if root is None:
        return 0

    left_depth: int = get_depth(root.left)
    right_depth: int = get_depth(root.right)
    return 1 + max(left_depth, right_depth)


def get_postfix_expr(root: Node) -> str:
    """
    Get the postfix expression of a tree.
    :param root:
    :return: postfix expression of the tree in the form of a string
    """
    postfix = []
    if root:
        postfix.extend(get_postfix_expr(root=root.left))
        postfix.extend(get_postfix_expr(root=root.right))
        postfix.append(str(root.value))
    return ' '.join(postfix)


def count_nodes(root: Node) -> int:
    """
    Count the number of nodes in a tree.
    :param root:
    :return: number of nodes in the tree
    """
    return 0 if root is None else 1 + count_nodes(root=root.left) + count_nodes(root=root.right)


def clone(root: Node) -> Optional[Node]:
    """
    Clone a tree.
    :param root:
    :return: cloned tree
    """
    if root is None:
        return None

    return Node(root.value, clone(root=root.left), clone(root=root.right))


def get_node_level(root: Node, index: int) -> int:
    """
    Get the level of a node in a tree.
    :param root:
    :param index:
    :return:
    """
    queue = deque([(root, 1)])
    count = 0
    level = 1

    while queue:
        node, current_level = queue.popleft()

        if count == index:
            return level

        if node.left:
            queue.append((node.left, current_level + 1))
            count += 1

        if count == index:
            level = current_level + 1
            return level

        if node.right:
            queue.append((node.right, current_level + 1))
            count += 1

        if count == index:
            level = current_level + 1
            return level

    return -1


def get_node(root: Node, index: int) -> Optional[Node]:
    if root is None:
        return None

    height = get_depth(root)
    max_index = 2 ** height - 1
    current_level = [root]
    level_index = 0

    while current_level:
        node = current_level[level_index]

        if level_index == index:
            return node

        if node.left is not None:
            left_index = 2 * level_index + 1

            if left_index <= max_index:
                current_level.append(node.left)

        if node.right is not None:
            right_index = 2 * level_index + 2

            if right_index <= max_index:
                current_level.append(node.right)

        level_index += 1

        if level_index == len(current_level):
            current_level = current_level[level_index:]
            level_index = 0

    return None
