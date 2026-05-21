from collections import deque

def breadth_first_search(graph, origin, destinations):

    # fifo queue for frontier
    # (current_node, path_to_node)
    frontier = deque()

    # visited nodes to prevent re-expansion
    visited = set()

    # parent relationships, useful for debugging
    parent = {origin: None}

    # number of nodes generated
    number_of_nodes = 1

    # intialise frontier
    frontier.append((origin, [origin]))

    # Bmain loop, continue until no nodes left to explore
    while frontier:
        # Pop from the left
        current_node, path = frontier.popleft()

        # Skip if already visited
        if current_node in visited:
            continue

        # Mark node as visited
        visited.add(current_node)

        # check if current node is a destination
        if current_node in destinations:
            return current_node, number_of_nodes, path

        # tie-breaking
        neighbours = sorted(graph[current_node], key=lambda x: x[0])

        # Expand neighbours
        for neighbour, cost in neighbours:
            # Only add unvisited and unseen nodes
            if neighbour not in visited and neighbour not in parent:
                parent[neighbour] = current_node

                # new path including this neighbour
                new_path = path + [neighbour]

                # add to frontier
                frontier.append((neighbour, new_path))

                # increment node counter
                number_of_nodes += 1

    # no goal found
    return None, number_of_nodes, []
