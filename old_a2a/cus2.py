import math


# straight-line distance between two nodes
def euclidean_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# find the longest geometric edge in the graph for heuristic
def max_edge_length(graph, nodes):
    longest = 0

    for current in graph:
        for neighbour, cost in graph[current]:
            length = euclidean_distance(nodes[current], nodes[neighbour])
            if length > longest:
                longest = length

    # avoid dividing by 0
    if longest == 0:
        return 1

    return longest


# estimate how many moves left to the nearest goal
def heuristic(node, nodes, destinations, longest_edge):
    valid_goals = [goal for goal in destinations if goal in nodes]

    if not valid_goals: #checking for no destination cases
        return 0

    best_distance = float("inf")

    for goal in valid_goals:
        distance = euclidean_distance(nodes[node], nodes[goal])
        if distance < best_distance:
            best_distance = distance

    if best_distance == float("inf"):
        return 0

    return math.ceil(best_distance / longest_edge)


def custom_search_2(graph, nodes, origin, destinations):
    destinations = {d for d in destinations if d in nodes}

    if origin not in nodes or not destinations:
        return None, 0, []
    
    visited_path = {origin}      # nodes on current DFS branch
    path = [origin]              # current path
    number_of_nodes = 1          # count created nodes
    longest_edge = max_edge_length(graph, nodes)

    # recursive IDA* search function
    def search(g, threshold):
        nonlocal number_of_nodes

        current = path[-1]
        h = heuristic(current, nodes, destinations, longest_edge)
        f = g + h

        # stop exploring if over current threshold
        if f > threshold:
            return f

        # goal test
        if current in destinations:
            return "FOUND"
        
        smallest_exceeded = float("inf") 

        neighbours = sorted(graph[current], key=lambda x: x[0])

        for neighbour, cost in neighbours:
            if neighbour in visited_path:
                continue

            path.append(neighbour)
            visited_path.add(neighbour)
            number_of_nodes += 1

            # CUS2: every move costs 1
            result = search(g + 1, threshold)

            if result == "FOUND":
                return "FOUND"

            if result < smallest_exceeded:
                smallest_exceeded = result #find the minimum of all ‘f’ greater than threshold encountered 

            path.pop()
            visited_path.remove(neighbour)

        return smallest_exceeded

    # initial threshold = h(start)
    threshold = heuristic(origin, nodes, destinations, longest_edge)

    while True:
        result = search(0, threshold)

        if result == "FOUND":
            return path[-1], number_of_nodes, path[:]

        if result == float("inf"):
            return None, number_of_nodes, []

        threshold = result