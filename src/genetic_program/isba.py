from typing import List

from tree.node import Node


def calculate_gsim(individual: Node, local_optima: List[Node], rthresh: float, gthresh: float, d: int):
    if len(local_optima) == 0:
        return False

    root_count = sum([1 for local_optimum in local_optima if local_optimum.value == individual.value])
    if root_count / len(local_optima) > rthresh:
        return True

    def compare_trees_up_to_depth_d(first_tree: Node, second_tree: Node, depth: int):
        if not first_tree or not second_tree:
            return True

        if not first_tree or not second_tree:
            return False

        if depth == 0:
            return first_tree.value == second_tree.value

        left_equal = compare_trees_up_to_depth_d(
            first_tree=first_tree.left,
            second_tree=second_tree.left,
            depth=depth - 1
        )
        right_equal = compare_trees_up_to_depth_d(
            first_tree=first_tree.right,
            second_tree=second_tree.right,
            depth=depth - 1
        )
        return left_equal and right_equal

    component_count = sum([1 for local_optimum in local_optima if
                           compare_trees_up_to_depth_d(first_tree=individual, second_tree=local_optimum, depth=d)])
    return component_count / len(local_optima) > gthresh


def get_relations(individual: Node):
    relations = []

    if individual is None:
        return relations

    def traverse_and_extract_relations(node: Node):
        if node is None:
            return None, None, None

        function_node = node.value
        left_child_function, left_child_left, left_child_right = traverse_and_extract_relations(node=node.left)
        right_child_function, right_child_left, right_child_right = traverse_and_extract_relations(node=node.right)

        relations.append((function_node, left_child_function, right_child_function))
        return function_node, node.left, node.right

    traverse_and_extract_relations(node=individual)
    return relations


def calculate_lsim(first_individual: Node, second_individual: Node, lthresh: int):
    first_relations = get_relations(individual=first_individual)
    second_relations = get_relations(individual=second_individual)

    intersection = set(first_relations).intersection(set(second_relations))
    return len(intersection) > lthresh
